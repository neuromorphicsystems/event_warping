import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import event_warping
import json
from matplotlib import mlab 
import matplotlib as mpl

TMAX                = 100e6
METHOD              = ["L-BFGS-B","BFGS", "Nelder-Mead"]
HEURISTIC           = ["variance","weighted_variance"]
RESOLUTION          = 100
vx_gt               = 20.79
vy_gt               = -2.57
np.seterr(divide='ignore', invalid='ignore')
FILENAME = "20220125_Mexican_Coast_205735_2022-01-25_20~58~52_NADIR.h5.es" #"20220125_Mexican_Coast_205735_2022-01-25_20~58~52_NADIR.h5.es"
width, height, events = event_warping.read_es_file("/media/sam/Samsung_T51/PhD/Code/orbital_localisation/data/es/NADIR/" + FILENAME)
width+=1
height+=1
events = event_warping.without_most_active_pixels(events, ratio=0.000001)
ii = np.where(np.logical_and(events["t"]>1, events["t"]<(TMAX)))
events = events[ii]
print(f"{len(events)=}")
deltaT = (events["t"][-1]-events["t"][0])/1e6
INITIAL_VELOCITY    = (20 / 1e6, -5 / 1e6)

#### MAKE FIGURE
# with open("/media/sam/Samsung_T51/PhD/Code/orbital_localisation/test_files/FN034HRTEgyptB_NADIR.es_weightVar_-50_50_100_1e-05_0.json") as input:
#     pixels = np.reshape(json.load(input), (RESOLUTION, RESOLUTION))
with open("/media/sam/Samsung_T51/PhD/Code/denseVar/img/20220125_Mexican_Coast_205735_2022-01-25_20~58~52_NADIR.h5.es_weightVar_-50_50_100_1e-05_1.json") as input:
    pixels = np.reshape(json.load(input), (RESOLUTION, RESOLUTION))

x = np.tile(np.arange(-pixels.shape[1], pixels.shape[1],2), (pixels.shape[0], 1))
y = np.tile(np.arange(-pixels.shape[1], pixels.shape[0],2), (pixels.shape[1], 1)).T

def f(x,y):
    "Objective function"
    return -event_warping.intensity_weighted_variance((width, height),events,velocity=(x, y))
 
# Hyper-parameter of the algorithm
c1 = c2 = 0.1
w = 0.5
 
# Create particles
n_particles = 100
np.random.seed(RESOLUTION)
X = (2 * np.random.rand(2,n_particles) - 1)*RESOLUTION
V = np.random.randn(2, n_particles) * RESOLUTION/10

# Initialize data
pbest = X
pbest_obj = []
for idx in range(X.shape[1]):
    variance = f(X[0][idx]/ 1e6, X[1][idx]/ 1e6)
    pbest_obj.append(variance)
pbest_obj = np.array(pbest_obj)
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()

def update():
    "Function to do one iteration of particle swarm optimization"
    global V, X, pbest, pbest_obj, gbest, gbest_obj
    # Update params
    r1, r2 = np.random.rand(2)
    V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
    X = X + V
    obj = []
    for idx in range(X.shape[1]):
        variance = f(X[0][idx]/ 1e6, X[1][idx]/ 1e6)
        obj.append(variance)
    obj = np.array(obj)
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

# Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(-pixels, extent=[-RESOLUTION, RESOLUTION, -RESOLUTION, RESOLUTION], origin='lower', cmap='viridis', alpha=0.5,label='_nolegend_')
sm = plt.cm.ScalarMappable(cmap='RdBu_r')
CS2 = plt.contourf(x, y, -pixels,300,alpha=0.4, cmap='RdBu_r',label='_nolegend_')
contours = plt.contour(CS2,alpha=0.4, cmap='RdBu_r',label='_nolegend_')
pbest_plot = ax.scatter(pbest[0], pbest[1]-7, marker='o', color='black', alpha=0.5,label='_nolegend_')
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5,label='_nolegend_')
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([-1], [-0.9], marker='*', s=100, color='yellow', alpha=0,label='_nolegend_')
ax.set_xlim([-RESOLUTION, RESOLUTION])
ax.set_ylim([-RESOLUTION, RESOLUTION])
ax.set_xlabel(r"$v_x$ [px/s]", fontsize=20)
ax.set_ylabel(r"$v_y$ [px/s]", fontsize=20)
ax.set_title(r"$Mexico - Iteration: 20/20$", fontsize=25)
font = {'family': 'latex','color':  'white','weight': 'normal','size': 18}
# plt.text(-15, -20, r'$Global \ Maxima$', fontdict=font,label='_nolegend_')
plt.text(33, -15, r'$True \ \theta$', fontdict=font,label='_nolegend_')
cmap = mpl.cm.RdBu
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap)).set_label(label=r"$Variance$",size=20)
# plt.axis([-50, 50, -30, 30])
x_ticks = [-100, 0, 100]# x_ticks = [-50, 0, 50]
x_labels = ['-100', '0', '100']#x_labels = ['-50', '0', '50']
plt.xticks(ticks=x_ticks, labels=x_labels)
y_ticks = [-100, 0, 100]#y_ticks = [-30, 0, 30]
y_labels = ['-100', '0', '100']#y_labels = ['-30', '0', '30']
plt.yticks(ticks=y_ticks, labels=y_labels)
# plt.show()

def animate(i):
    "Steps of PSO: algorithm update and show in plot"
    print("Iteration: {}".format(i))
    title = r"$Mexico - Iteration: {:02d}/30$".format(i)
    update()
    ax.set_title(title, fontsize=20)
    pbest_plot.set_offsets(pbest.T)
    # X[0]+=60
    # X[1]-=0
    p_plot.set_offsets(X.T)
    p_arrow.set_offsets(X.T)
    p_arrow.set_UVC(V[0], V[1])
    gbest_plot.set_offsets(gbest.reshape(1,-1))
    plt.savefig("img/weight_map/"+FILENAME +"_"+str(i)+".svg", bbox_inches = 'tight')
    return ax, pbest_plot, p_plot, p_arrow, gbest_plot

anim = FuncAnimation(fig, animate, frames=list(range(1,30)), interval=500, blit=False, repeat=True)
anim.save("./img/"+FILENAME+".gif", dpi=120, writer="imagemagick")
print("PSO found best solution at f({})={}".format(gbest, gbest_obj))