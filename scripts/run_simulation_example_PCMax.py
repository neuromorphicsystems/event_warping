import json,os,sys,pathlib,skimage,PIL.Image,itertools,event_warping,concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from tqdm import tqdm
from scipy import ndimage

np.seterr(divide='ignore', invalid='ignore')
variance_loss       = []
fp                  = 0
fp1                 = 0
counting            = 0
fun_idx             = 0
SAVE                = 1
WEIGHT              = 1
VIDEOS              =[ "equally_fired_pixels"]
OBJECTIVE           = ["variance","weighted_variance"]
FILENAME            = VIDEOS[0]
HEURISTIC           = OBJECTIVE[WEIGHT]
VELOCITY_RANGE      = (-50,50)
RESOLUTION          = 20
TMAX                = 10e6
RATIO               = 0.00001
imgpath             = "/media/sam/Samsung_T51/PhD/Code/orbital_localisation/img/weight_map"
READFROM            = "/media/sam/Samsung_T51/PhD/Code/orbital_localisation/test_files/"
SAVEFILETO          = "/media/sam/Samsung_T51/PhD/Code/orbital_localisation/img/"
scalar_velocities   = np.linspace(VELOCITY_RANGE[0],VELOCITY_RANGE[1],RESOLUTION)
nVel                = len(scalar_velocities)

def deleteimg():
    # Check if the folder exists
    if os.path.exists(imgpath):
        # Get a list of all files in the folder
        files = os.listdir(imgpath)
        if len(files) > 0:
            # Iterate through the files and delete them
            for file in files:
                file_path = os.path.join(imgpath, file)
                os.remove(file_path)
            print("All files in the folder have been deleted.")
        else:
            print("No images found in this folder.")
    else:
        print("The specified folder does not exist.")

def without_most_active_pixels(events: np.ndarray, ratio: float):
    assert ratio >= 0.0 and ratio <= 1.0
    count = np.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    np.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    return events[
        count[events["x"], events["y"]]
        <= np.percentile(count, 100.0 * (1.0 - ratio))
    ]

# events = event_warping.read_h5_file(READFROM + FILENAME)
# width, height= 241, 181
width, height, events = event_warping.read_es_file(READFROM + FILENAME + ".es")
# width+=1
# height+=1
events = without_most_active_pixels(events, ratio=0.000001)
TMAX = events["t"][-1]
ii = np.where(np.logical_and(events["t"]>1, events["t"]<(TMAX)))
events = events[ii]
t = (events["t"][-1]-events["t"][0])/1e6
EDGEPX = t
print(f"{len(events)=}")

def weight_f1(vx,vy,remove_edge_pixels=False):
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] = eventmap.pixels[i+1,j+1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 3:
    [i,j] = np.where((x >= vx) & (x <= width) & (y > 0) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 4:
    [i,j] = np.where((x >= width) & (x <= width+vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 5:
    [i,j] = np.where((x > vx) & (x < width) & (y > height) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 6:
    [i,j] = np.where((x > 0) & (x < vx) & (y > 0) & (y < ((vy*x)/(vx))))
    eventmap.pixels[i+1,j+1] *= (vy/y[i,j])
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= ((vy*x)/(vx))) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 8:
    [i,j] = np.where((x >= width) & (x <= width+vx) & (y < height+vy) & (y > (((vy*(x-width))/(vx))+height)))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 9:
    [i,j] = np.where((x >= width) & (x <= width+vx) & (y > height) & (y <= (((vy*(x-width))/(vx))+height)))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 10:
    [i,j] = np.where((x > width) & (x < width+vx) & (y >= ((vy*(x-width))/(vx))) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x > 0) & (x < vx) & (y > height) & (y <= (((vy*x)/(vx))+height)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*height-vx*y[i,j]+vy*x[i,j])
    
    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                = 0
        eventmap.pixels[x < EDGEPX]                         = 0
        eventmap.pixels[y > height+vy-EDGEPX]               = 0
        eventmap.pixels[y < EDGEPX]                         = 0
        eventmap.pixels[y < ((vy*(x-width))/(vx))+EDGEPX]   = 0
        eventmap.pixels[y > (((vy*x)/(vx))+height)-EDGEPX]  = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]              = 0
    return eventmap.pixels

def weight_f2(vx,vy,remove_edge_pixels=False):
    # Condition 1:
    [i,j] = np.where((x > 0) & (x < width) & (y >= (vy*x)/vx) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    middlePlan = (vx)/x[i,j]
    # Condition 2:
    [i,j] = np.where((x >= width) & (x < vx) & (y >= vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 3:
    [i,j] = np.where((x >= width) & (x <= vx) & (y >= (vy*x)/vx) & (y < vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 4:
    [i,j] = np.where((x >= width) & (x < vx) & (y >= height) & (y <= (vy/vx)*(x-width-vx)+vy+height))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 5:
    [i,j] = np.where((x > 0) & (x <= width) & (y > 0) & (y < (vy*x)/vx))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 6:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y >= height) & (y <= (vy/vx)*(x-width-vx)+vy+height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 7:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width-vx)+vy+height) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 8:
    [i,j] = np.where((x > 0) & (x < width) & (y > vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 9:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 10:
    [i,j] = np.where((x > width) & (x < vx) & (y <= (vy*x)/vx) & (y > 0))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x >= vx) & (x <= vx+width) & (y >= (vy*(x-width))/vx) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 12:
    [i,j] = np.where((x > width) & (x < vx) & (y > (vy/vx)*(x-width-vx)+vy+height) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i-1,j-1]+vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > 0) & (x <= width) & (y < (vy/vx)*x+height) & (y >= height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i-1,j-1]+vy*x[i-1,j-1])
    
    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]                 = 0
        eventmap.pixels[y < (vy*(x-width))/vx+EDGEPX]                = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f3(vx,vy,remove_edge_pixels=False):
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] = eventmap.pixels[i+1,j+1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 3:
    [i,j] = np.where((x > vx) & (x < width) & (y >= 0) & (y <= vy))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 4:
    [i,j] = np.where((x > vx) & (x < width) & (y >= height) & (y <= height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i-1,j-1]+height+vy)
    # Condition 5:
    [i,j] = np.where((x > width) & (x < width+vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i-1,j-1]+width+vx)
    # Condition 6:
    [i,j] = np.where((x > 0) & (x < vx) & (y > height) & (y <= ((-vy*x)/vx)+height+vy))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i-1,j-1]
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= ((-vy*x)/vx)+height+vy) & (y <= height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 8:
    [i,j] = np.where((x > width) & (x < width+vx) & (y >= ((vy*(-x+width+vx))/vx)) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i-1,j-1]+width+vx)
    # Condition 9:
    [i,j] = np.where((x >= width) & (x < width+vx) & (y >= 0) & (y <= ((vy*(-x+width+vx))/vx)))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 10:
    [i,j] = np.where((x > 0) & (x <= vx) & (y > ((-vy*(x))/vx)+vy) & (y <= vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x >= width) & (x <= width+vx) & (y > height) & (y < ((vy*(-x+width+vx))/vx)+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    
    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                        = 0
        eventmap.pixels[x < EDGEPX]                                 = 0
        eventmap.pixels[y > height+vy-EDGEPX]                       = 0
        eventmap.pixels[y < EDGEPX]                                 = 0
        eventmap.pixels[y < (((-vy*(x))/vx)+vy)+EDGEPX]             = 0
        eventmap.pixels[y > ((vy*(-x+width+vx))/vx)+height-EDGEPX]  = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                      = 0
    return eventmap.pixels

def weight_f4(vx,vy,remove_edge_pixels=False):
    # Condition 7:
    [i,j] = np.where((x > vx) & (x < width) & (y >= 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    middlePlan = (vy)/y[i,j]
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y >= height) & (y <= vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= height) & (y < (vy*x)/vx))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 3:
    [i,j] = np.where((x >= width) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 4:
    [i,j] = np.where((x > width) & (x < vx+width) & (y <= ((vy*(x-width))/vx)+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i-1,j-1]+width+vx)
    # Condition 6:
    [i,j] = np.where((x >= width) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i-1,j-1]+height+vy)
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < vx) & (y > 0) & (y < (vy*x)/vx) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 8:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= (vy*x)/vx) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 9:
    [i,j] = np.where((x >= vx) & (x < width) & (y > vy) & (y <= vy+height))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i-1,j-1]+height+vy)
    # Condition 10:
    [i,j] = np.where((x > 0) & (x <= vx) & (y > (vy/vx)*x) & (y < (vy/vx)*x+height) & (y >= height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x > 0) & (x < vx) & (y < (vy/vx)*x+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 12:
    [i,j] = np.where((x >= width) & (x <= vx+width) & (y <= height) & (y > (vy/vx)*(x-width)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > width) & (x < vx+width) & (y < ((vy*(x-width))/vx)+height) & (y > height) & (y <vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]            = 0
        eventmap.pixels[x < EDGEPX]                     = 0
        eventmap.pixels[y > height+vy-EDGEPX]           = 0
        eventmap.pixels[y < EDGEPX]                     = 0
        eventmap.pixels[y < (vy/vx)*(x-width)+EDGEPX]   = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]          = 0
    return eventmap.pixels

def weight_f5(vx,vy,remove_edge_pixels=False):
    # Condition 2:
    [i,j] = np.where((x > 0) & (x < width) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(x[i,j])
    middlePlan = (vx)/(x[i,j])
    # Condition 1:
    [i,j] = np.where((x >= width) & (x <= vx) & (y > (vy/vx)*(width-x)+vy) & (y < ((-vy*x)/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 3:
    [i,j] = np.where((x > vx) & (x < width+vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 4:
    [i,j] = np.where((x > 0) & (x < width) & (y > ((-vy*x)/(vx))+height+vy) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 5:
    [i,j] =np.where((x > 0) & (x < width) & (y > height) & (y < ((-vy*x)/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 6:
    [i,j] =np.where((x > vx) & (x < vx+width) & (y > (vy*(-x+width+vx))/(vx)) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 7:
    [i,j] =np.where((x >= vx) & (x < vx+width) & (y > 0) & (y < (vy*(-x+width+vx))/(vx)))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 8:
    [i,j] =np.where((x >= width) & (x <= vx) & (y > ((-vy*x)/(vx))+height+vy) & (y < ((vy*(-x+width))/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= (vy*vx)/(height*vx+vx*vy-vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 9:
    [i,j] =np.where((x >= vx) & (x < vx+width) & (y > height) & (y < ((vy*(-x+width))/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= (vy*vx)/(height*vx+vx*vy-vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 10:
    [i,j] =np.where((x >= 0) & (x < width) & (y > (-vy/vx)*x+vy) & (y < vy))
    eventmap.pixels[i+1,j+1] *= -(vy*vx)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 11:
    [i,j] =np.where((x >= width) & (x < vx) & (y > (-vy/vx)*x+vy) & (y < (vy/vx)*(width-x)+vy))
    eventmap.pixels[i+1,j+1] *= -(vy*vx)/(vx*vy-vx*y[i,j]-vy*x[i,j])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > ((vy*(-x+width))/(vx))+height+vy-EDGEPX] = 0
        eventmap.pixels[y < (-vy/vx)*x+vy+EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f6(vx,vy,remove_edge_pixels=False):
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y > 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vy)/(y[i,j])
    middlePlan = (vy)/(y[i,j])
    # Condition 2:
    [i,j] = np.where((x > vx) & (x < width) & (y >= height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 3:
    [i,j] = np.where((x > 0) & (x < vx) & (y > (-vy/vx)*x+vy+height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 4:
    [i,j] = np.where((x >= width) & (x <= vx+width) & (y > height) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 5:
    [i,j] = np.where((x > vx) & (x < width) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 6:
    [i,j] = np.where((x > 0) & (x < vx) & (y <= (-vy/vx)*x+vy+height) & (y >= (-vy/vx)*x+vy) & (y > height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 7:
    [i,j] = np.where((x > 0) & (x <= vx) & (y >= (-vy/vx)*x+vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 8:
    [i,j] = np.where((x > 0) & (x <= vx) & (y >= vy) & (y < (-vy/vx)*x+vy+height))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 9:
    [i,j] = np.where((x > 0) & (x <= vx) & (y > (-vy/vx)*x+vy+height) & (y <= vy+height) & (y >= vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 10:
    [i,j] = np.where((x >= width) & (x < vx+width) & (y > (vy/vx)*(-x+width)+vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i-1,j-1]+width+vx)
    # Condition 11:
    [i,j] = np.where((x >= width) & (x < vx+width) & (y > 0) & (y < (vy/vx)*(-x+width)+vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    # Condition 12:
    [i,j] = np.where((x >= width) & (x <=vx+width) & (y < (vy/vx)*(-x+width)+vy+height) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width)+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x >= width) & (x <= vx+width) & (y < vy+height) & (y > (vy/vx)*(-x+width)+vy) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > (vy/vx)*(-x+width)+vy+height-EDGEPX]     = 0
        eventmap.pixels[y < (-vy/vx)*x+vy+EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f7(vx,vy,remove_edge_pixels=False):
    # Condition 4:
    [i,j] = np.where((x > 0) & (x < width) & (y > 0) & (y <= (vy/vx)*x) & (y < height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    middlePlan = vy/y[i,j]
    # Condition 1:
    [i,j] = np.where((x > width) & (x < vx) & (y > (vy/vx)*(x-width)+height) & (y < (vy/vx)*x))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x <= width) & (y >= height) & (y < (vy/vx)*x))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 3:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)+height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 5:
    [i,j] = np.where((x > 0) & (x < width) & (y > (vy/vx)*x) & (y <= height))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 6:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)+height) & (y <= vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 7:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y <= (vy/vx)*(x-width)+height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 8:
    [i,j] = np.where((x >= width) & (x < vx) & (y > (vy/vx)*(x-width)) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 9:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 10:
    [i,j] = np.where((x >= width) & (x < vx) & (y > -(vy/vx)*x+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x > 0) & (x < width) & (y > -(vy/vx)*x+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 12:
    [i,j] = np.where((x > 0) & (x < vx) & (y > (vy/vx)*x) & (y < vy) & (y > height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 13:
    [i,j] = np.where((x > width) & (x < vx+width) & (y < (vy/vx)*(x-width)+height) & (y > height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y < (vy/vx)*(x-width)+EDGEPX]                = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]                 = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f8(vx,vy,remove_edge_pixels=False):
    # Condition 1:
    [i,j] = np.where((x > 0) & (x < width) & (y > 0) & (y < (vy/vx)*x) & (y > 0))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    middlePlan = vy/y[i,j]
    # Condition 2:
    [i,j] = np.where((x > width) & (x <= vx) & (y <= (vy/vx)*(x-width)+height) & (y >= (vy/vx)*x) & (y > height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 3:
    [i,j] = np.where((x >= width) & (x <= vx) & (y <= (vy/vx)*(x-width)+height) & (y >= (vy/vx)*x) & (y > 0) & (y <= height))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 4:
    [i,j] = np.where((x > width) & (x <= vx) & (y <= (vy/vx)*(x-width)+height) & (y >= (vy/vx)*x) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 5:
    [i,j] = np.where((x > 0) & (x < width) & (y >= (vy/vx)*x) & (y <= height))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 6:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)+height) & (y <= vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 7:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y <= (vy/vx)*(x-width)+height) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 8:
    [i,j] = np.where((x >= width) & (x < vx) & (y < (vy/vx)*x) & (y >= 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 9:
    [i,j] = np.where((x >= width) & (x < vx) & (y < (vy/vx)*x) & (y >= height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 10:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y < (vy/vx)*x) & (y >= 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 11:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y < (vy/vx)*x) & (y >= height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 12:
    [i,j] = np.where((x >= width) & (x < vx) & (y > (vy/vx)*(x-width)+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 13:
    [i,j] = np.where((x >= width) & (x < vx) & (y > (vy/vx)*(x-width)+height) & (y < vy) & (y > height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 14:
    [i,j] = np.where((x >= 0) & (x < width) & (y > (vy/vx)*(x-width)+height) & (y < vy) & (y > height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 15:
    [i,j] = np.where((x >= 0) & (x < width) & (y > (vy/vx)*(x-width)+height) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y < (vy/vx)*(x-width)+EDGEPX]                = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]                 = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f9(vx,vy,remove_edge_pixels=False):
    # Condition 1:
    [i,j] = np.where((x > 0) & (x < width) & (y > (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    middlePlan = (vy)/(-y[i,j]+height+vy)
    # Condition 2:
    [i,j] = np.where((x > width) & (x < vx) & (y > (-vy/vx)*x+height+vy) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= middlePlan[0]
    # Condition 3:
    [i,j] = np.where((x > 0) & (x <= width) & (y > (-vy/vx)*x+height+vy) & (y <= vy))
    eventmap.pixels[i+1,j+1] *= middlePlan[0]
    # Condition 4:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y >= height) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= middlePlan[0]
    # Condition 5:
    [i,j] = np.where((x > 0) & (x < width) & (y < (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 6:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y < (vy/vx)*(-x+width+vx)) & (y > 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    # Condition 5:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y > (vy/vx)*(-x+width+vx)) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 6:
    [i,j] = np.where((x >= width) & (x <= vx) & (y <= height) & (y > 0))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < width) & (y < height) & (y > 0))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 8:
    [i,j] = np.where((x > 0) & (x <= width) & (y >= height) & (y <= vy) & (y < (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 9:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y < (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 10:
    [i,j] = np.where((x >= width) & (x <= vx) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 11:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 12:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > (vy/vx)*(-x+width)+vy+height-EDGEPX]     = 0
        eventmap.pixels[y < (-vy/vx)*x+vy+EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f10(vx,vy,remove_edge_pixels=False):
    # Condition 6:
    [i,j] = np.where((x > width) & (x < vx) & (y <= height) & (y > 0) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    middlePlan = -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 1:
    [i,j] = np.where((x > width) & (x < vx) & (y < (-vy/vx)*x+height+vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= middlePlan[-1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x <= width) & (y > (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 3:
    [i,j] = np.where((x > 0) & (x <= width) & (y < (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 4:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y < (vy/vx)*(-x+width+vx)) & (y > 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    # Condition 5:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y > (vy/vx)*(-x+width+vx)) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 7:
    [i,j] = np.where((x > 0) & (x <= width) & (y > 0) & (y <= height))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 8:
    [i,j] = np.where((x > 0) & (x <= width) & (y > height) & (y < vy) & (y < (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 9:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 10:
    [i,j] = np.where((x > width) & (x < vx) & (y > vy) & (y < vy+height) & (y > (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 11:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y > (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 12:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y < (vy/vx)*(-x+width)+vy-EDGEPX]            = 0
        eventmap.pixels[y > (-vy/-x)*x+vy-EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def saveimg(contrastmap,var,fun_idx,draw=False):
    colormap = plt.get_cmap("magma")
    gamma = lambda image: image ** (1 / 1)
    scaled_pixels = gamma((contrastmap - contrastmap.min()) / (contrastmap.max() - contrastmap.min()))
    image = PIL.Image.fromarray((colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8))
    d = ImageDraw.Draw(image)
    # d.text((10,10), "f_" + str(fun_idx) + "(x,y)", fill=(0,255,0))
    if draw==True:
        d.line((0,0, 0,height+vy), fill=128, width = 1)
        d.line((0,0, width+vx,0),  fill=128, width = 1)
        d.text((2,0), "0", fill=(0,255,0))
        d.line((0,height+vy, width+vx,height+vy), fill=128, width = 1)
        d.text((2,height+vy-10), "h+vy*t", fill=(0,255,0))
        d.line((0,height, width+vx,height), fill=128, width = 1)
        d.text((2,height), "h", fill=(0,255,0))
        d.line((0,vy, width+vx,vy), fill=128, width = 1)
        d.text((2,vy), "vy*t", fill=(0,255,0))
        d.line((vx,0, vx,height+vy), fill=128, width = 1)
        d.text((vx-25,0), "vx*t", fill=(0,255,0))
        d.line((width,0, width,height+vy), fill=128, width = 1)
        d.text((width-10,0), "w", fill=(0,255,0))
        d.line((vx+width,0, vx+width,height+vy), fill=128, width = 1)
        d.text((vx+width-40,0), "w+vx*t", fill=(0,255,0))
    new_image = image.resize((500, 500))
    im1 = new_image.save("img/weight_map/f_" + str(fun_idx) +"_vy_" + str(fieldy*t)  + "_vx_" + str(fieldx*t) + "_" + "var_" + str(var) + ".jpg")
    fun_idx=0

def varianceObj(data):
    n = len(data)
    mean = sum(data) / n
    squared_differences = [(x - mean)**2 for x in data]
    variance = sum(squared_differences) / (n - 1)
    return variance

def variance(evmap):
    flatimg = evmap.flatten()
    res = flatimg[flatimg != 0]
    return np.var(res)

if SAVE:
    deleteimg()
for iVely in range(0,nVel):
    fp+=1
    fieldy = scalar_velocities[iVely]
    for iVelx in range(0,nVel):
        fp1+=1
        counting += 1
        fieldx = scalar_velocities[iVelx]
        velocity = (fieldx * 1e-6, fieldy * 1e-6)
        eventmap = event_warping.accumulate((width, height), events, velocity)
        vx=np.abs(fieldx*t)
        vy=np.abs(fieldy*t)
        
        x = np.tile(np.arange(1, eventmap.pixels.shape[1]+1), (eventmap.pixels.shape[0], 1))
        y = np.tile(np.arange(1, eventmap.pixels.shape[0]+1), (eventmap.pixels.shape[1], 1)).T

        if WEIGHT:
            #f_1(x,y)
            if fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height:
                fun_idx                 += 1
                evmap                   = weight_f1(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0
            
            #f_2(x,y)
            if fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height:
                fun_idx                 += 2
                evmap                   = weight_f2(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_3(x,y)
            if fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height:
                fun_idx                 += 3
                evmap                   = weight_f3(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_4(x,y)
            if fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height or fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height:
                fun_idx                 += 4
                evmap                   = weight_f4(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_5(x,y)
            if fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height:
                fun_idx                 += 5
                evmap                   = weight_f5(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_6(x,y)
            if fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height or fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height:
                fun_idx                 += 6
                evmap                   = weight_f6(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_7(x,y)
            if (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
                fun_idx                 += 7
                evmap                   = weight_f7(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_8(x,y)
            if  (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
                fun_idx                 += 8
                evmap                   = weight_f8(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_9(x,y)
            if (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
                fun_idx                 += 9
                evmap                   = weight_f9(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0

            #f_10(x,y)
            if  (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
                fun_idx                 += 10
                evmap                   = weight_f10(vx,vy,remove_edge_pixels=True)
                var                     = variance(evmap)
                saveimg(evmap,var,fun_idx,draw=False)
                fun_idx=0
            else:
                var = variance(eventmap.pixels)
        else:
            var = variance(eventmap.pixels)

        variance_loss.append(var)
        sys.stdout.write(f"\r{len(variance_loss)} / {RESOLUTION ** 2} ({(len(variance_loss) / (RESOLUTION ** 2) * 100):.2f} %)")
        sys.stdout.flush()
    fp1=0
sys.stdout.write("\n")
with open(
        f"{SAVEFILETO + FILENAME}_{HEURISTIC}_{VELOCITY_RANGE[0]}_{VELOCITY_RANGE[1]}_{RESOLUTION}_{RATIO}_{WEIGHT}.json",
        "w",
    ) as output:
        json.dump(variance_loss, output)

variance_loss = np.reshape(variance_loss, (RESOLUTION, RESOLUTION))
variance_loss[np.isnan(variance_loss)]=0
colormap = plt.get_cmap("magma")
gamma = lambda image: image ** (1 / 2)
scaled_pixels = gamma((variance_loss - variance_loss.min()) / (variance_loss.max() - variance_loss.min()))
image = PIL.Image.fromarray((colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8))
im1 = image.rotate(90, PIL.Image.NEAREST, expand = 1)
new_im1 = im1.resize((400, 400))

# dr = ImageDraw.Draw(new_im1)
# f1_pos = np.where(scalar_velocities*t<width)
# f1_neg = np.where(scalar_velocities*t<-width)
# dr.line((0,scalar_velocities[f1_pos[0][-1]+1]*t,new_im1.width,scalar_velocities[f1_pos[0][-1]+1]*t), fill=128, width = 1)
# dr.line((scalar_velocities[f1_pos[0][-1]+1]*t,0,scalar_velocities[f1_pos[0][-1]+1]*t,new_im1.width), fill=128, width = 1)
# dr.line((0,new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t,new_im1.width,new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t), fill=128, width = 1)
# dr.line((new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t,0,new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t,new_im1.width), fill=128, width = 1)
# dr.line((0,0,scalar_velocities[f1_pos[0][-1]+1]*t,scalar_velocities[f1_pos[0][-1]+1]*t), fill=128, width = 1)
# dr.line((0,new_im1.width,scalar_velocities[f1_pos[0][-1]+1]*t,new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t), fill=128, width = 1)
# dr.line((new_im1.width,0,new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t,scalar_velocities[f1_pos[0][-1]+1]*t), fill=128, width = 1)
# dr.line((new_im1.width,new_im1.width,new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t,new_im1.width-scalar_velocities[f1_pos[0][-1]+1]*t), fill=128, width = 1)
# dr.line((0,np.round(new_im1.width/2),new_im1.width,np.round(new_im1.width/2)), fill=128, width = 1)
# dr.line((np.round(new_im1.width/2),0,np.round(new_im1.width/2),new_im1.width), fill=128, width = 1)

im1 = new_im1.save(SAVEFILETO+FILENAME+"_"+str(VELOCITY_RANGE[0])+"_"+str(VELOCITY_RANGE[1])+"_"+str(RESOLUTION)+"_"+str(RATIO)+"_" +str(WEIGHT) +".png")
new_im1.show()
