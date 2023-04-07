import sympy as sp
import numpy as np
import PIL.Image
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

sp.init_printing()
 
#Declare symbolic variables explicitly which are to be treated as symbols
x,y,rho,vx,vy,w,h,t  = sp.symbols("x y rho vx vy w h t")

############# f_1(x,y) ####################
def fun1():
    f1_xy = [(rho*t),
            ((rho*x)/vx),
            ((rho*y)/vy),
            ((rho*(-x+w+vx*t))/vx),
            ((rho*(-y+h+vy*t))/vy),
            (rho*((-x/vx)+(y/vy)+(w/vx))),
            ((rho*((x/vx)-(y/vy)+(h/vy)))),
            ((rho*(x))/(vx)),
            ((rho*(y))/(vy)),
            ((rho*(-x+w+vx*t))/(vx)),
            ((rho*(-y+h+vy*t))/(vy))]

    limx_start_f1  = [vx*t,0,vx*t,w,vx*t,w,0,0,0,w,w]
    limx_finish_f1 = [w,vx*t, w, w+vx*t,w,w+vx*t,vx*t,vx*t,vx*t,w+vx*t,w+vx*t]
    limy_start_f1  = [vy*t, vy*t, 0,vy*t,h, (vy*(x-w))/vx,h, ((vy*x)/vx),0,h,(h-(((w-x)*vy)/vx))]
    limy_finish_f1 = [h,h, vy*t, h,h+vy*t,vy*t,((vy*x)/vx)+h,vy*t,((vy*x)/vx),(h-(((w-x)*vy)/vx)),h+vy*t]
    
    fbar_1 = (sp.integrate(f1_xy[0],  (y, limy_start_f1[0],  limy_finish_f1[0]),  (x, limx_start_f1[0],  limx_finish_f1[0]))+
              sp.integrate(f1_xy[1],  (y, limy_start_f1[1],  limy_finish_f1[1]),  (x, limx_start_f1[1],  limx_finish_f1[1]))+
              sp.integrate(f1_xy[2],  (y, limy_start_f1[2],  limy_finish_f1[2]),  (x, limx_start_f1[2],  limx_finish_f1[2]))+
              sp.integrate(f1_xy[3],  (y, limy_start_f1[3],  limy_finish_f1[3]),  (x, limx_start_f1[3],  limx_finish_f1[3]))+
              sp.integrate(f1_xy[4],  (y, limy_start_f1[4],  limy_finish_f1[4]),  (x, limx_start_f1[4],  limx_finish_f1[4]))+
              sp.integrate(f1_xy[5],  (y, limy_start_f1[5],  limy_finish_f1[5]),  (x, limx_start_f1[5],  limx_finish_f1[5]))+
              sp.integrate(f1_xy[6],  (y, limy_start_f1[6],  limy_finish_f1[6]),  (x, limx_start_f1[6],  limx_finish_f1[6]))+
              sp.integrate(f1_xy[7],  (y, limy_start_f1[7],  limy_finish_f1[7]),  (x, limx_start_f1[7],  limx_finish_f1[7]))+
              sp.integrate(f1_xy[8],  (y, limy_start_f1[8],  limy_finish_f1[8]),  (x, limx_start_f1[8],  limx_finish_f1[8]))+
              sp.integrate(f1_xy[9],  (y, limy_start_f1[9],  limy_finish_f1[9]),  (x, limx_start_f1[9],  limx_finish_f1[9]))+
              sp.integrate(f1_xy[10], (y, limy_start_f1[10], limy_finish_f1[10]), (x, limx_start_f1[10], limx_finish_f1[10])))/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))

    with open('./test_files/final_variance_equation_1.txt','wb') as f:
                for idx in tqdm(range(len(f1_xy))):
                        variance_f2 = sp.simplify(sp.integrate(((f1_xy[idx]-fbar_1)**2), (y, limy_start_f1[idx],  limy_finish_f1[idx]),  (x, limx_start_f1[idx],  limx_finish_f1[idx])))
                        pickle.dump(str(variance_f2),f)

############# f_2(x,y) ####################
def fun2():
        f2_xy = [(rho*t/vx),
                ((rho*x)/vx),
                ((rho*(-x+w+vx*t)/(vx))),
                (rho*(y/(vy*vx))),
                ((rho*(-y+h+vy*t))/(vx*vy)),
                (rho*(-(((x-vx*t)/w)-(y/vy))/vx)),
                ((rho*((t*x/(vx*w))-(y/(vx*vy))+(h/(vx*vy))))),
                ((rho*(y))/(vy*vx)),
                ((rho*(x))/(vx)),
                ((rho*(-x+w+vx*t))/(vx)),
                ((rho*(-y+h+vy*t))/(vx*vy))]

        limx_start_f2  = [w,0,vx*t,w,w,vx*t,0,0,0,vx*t,vx*t]
        limx_finish_f2 = [vx*t,w, w+vx*t, vx*t,vx*t,w+vx*t,w,w,w,w+vx*t,w+vx*t]
        limy_start_f2  = [vy*t,vy*t,vy*t,(vy/vx)*(x-w),h,(vy/vx)*(x-w),h,0,((vy*t*x)/w),h,((vy*t*x)/w)+h-(vx*vy*t**2)/w]
        limy_finish_f2 = [h,h,h,vy*t,(vy/vx)*x+h,vy*t,(vy/vx)*x+h,((vy*t*x)/w),vy*t,(((vy*t*x)/w)+h-(vx*vy*t**2)/w),h+vy*t]
        
        bot = 1/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))
        fbar_2 = bot*(sp.integrate(f2_xy[0],  (y, limy_start_f2[0],  limy_finish_f2[0]),  (x, limx_start_f2[0],  limx_finish_f2[0]))+
                sp.integrate(f2_xy[1],  (y, limy_start_f2[1],  limy_finish_f2[1]),  (x, limx_start_f2[1],  limx_finish_f2[1]))+
                sp.integrate(f2_xy[2],  (y, limy_start_f2[2],  limy_finish_f2[2]),  (x, limx_start_f2[2],  limx_finish_f2[2]))+
                sp.integrate(f2_xy[3],  (y, limy_start_f2[3],  limy_finish_f2[3]),  (x, limx_start_f2[3],  limx_finish_f2[3]))+
                sp.integrate(f2_xy[4],  (y, limy_start_f2[4],  limy_finish_f2[4]),  (x, limx_start_f2[4],  limx_finish_f2[4]))+
                sp.integrate(f2_xy[5],  (y, limy_start_f2[5],  limy_finish_f2[5]),  (x, limx_start_f2[5],  limx_finish_f2[5]))+
                sp.integrate(f2_xy[6],  (y, limy_start_f2[6],  limy_finish_f2[6]),  (x, limx_start_f2[6],  limx_finish_f2[6]))+
                sp.integrate(f2_xy[7],  (y, limy_start_f2[7],  limy_finish_f2[7]),  (x, limx_start_f2[7],  limx_finish_f2[7]))+
                sp.integrate(f2_xy[8],  (y, limy_start_f2[8],  limy_finish_f2[8]),  (x, limx_start_f2[8],  limx_finish_f2[8]))+
                sp.integrate(f2_xy[9],  (y, limy_start_f2[9],  limy_finish_f2[9]),  (x, limx_start_f2[9],  limx_finish_f2[9]))+
                sp.integrate(f2_xy[10], (y, limy_start_f2[10], limy_finish_f2[10]), (x, limx_start_f2[10], limx_finish_f2[10])
                ))
        fbar_2 = sp.simplify(f2_xy)

        with open('./test_files/final_variance_equation_2.txt','wb') as f:
                for idx in tqdm(range(len(f2_xy))):
                        variance_f2 = sp.simplify(sp.integrate(((f2_xy[idx]-fbar_2)**2), (y, limy_start_f2[idx],  limy_finish_f2[idx]),  (x, limx_start_f2[idx],  limx_finish_f2[idx])))
                        pickle.dump(str(variance_f2),f)

############# f_3(x,y) ####################
def fun3():
    f3_xy = [(rho*t),
            ((rho*x)/vx),
            ((rho*y)/vy),
            (rho*((-x+w+vx*t)/(vx))),
            ((rho*(-y+h+vy*t))/(vy)),
            (rho*(x*vy+y*vx-vx*vy*t)/(vx*vy)),
            ((rho*((h*vx)+(vx*vy*t)-(vx*y)+(vy*w)-vy*x)/(vx*vy))),
            ((rho*(y))/(vy)),
            ((rho*(-x+w+vx*t))/(vx)),
            ((rho*(x))/(vx)),
            ((rho*(-y+h+vy*t))/(vy))]

    limx_start_f3  = [vx*t,0,vx*t,w,vx*t,0,w,w,w,0,0]
    limx_finish_f3 = [w,vx*t,w,w+vx*t,w,vx*t,w+vx*t,w+vx*t,w+vx*t,vx*t,vx*t]
    limy_start_f3  = [vy*t,vy*t,0,vy*t,h,((-vy*x)/vx)+(vy*t),h,0,(((w+vx*t-x)*vy)/vx),h,h+vy*(t-(x/vx))]
    limy_finish_f3 = [h,h,vy*t,h,h+vy*t,vy*t,(((-vy*x)/vx)+((vy*w)/vx)+vy*t+h),(((w+vx*t-x)*vy)/vx),vy*t,h+vy*(t-(x/vx)),h+vy*t]

    bot = 1/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))
    fbar_3 =   bot*(sp.integrate(f3_xy[0],  (y, limy_start_f3[0],  limy_finish_f3[0]),  (x, limx_start_f3[0],  limx_finish_f3[0]))+
                    sp.integrate(f3_xy[1],  (y, limy_start_f3[1],  limy_finish_f3[1]),  (x, limx_start_f3[1],  limx_finish_f3[1]))+
                    sp.integrate(f3_xy[2],  (y, limy_start_f3[2],  limy_finish_f3[2]),  (x, limx_start_f3[2],  limx_finish_f3[2]))+
                    sp.integrate(f3_xy[3],  (y, limy_start_f3[3],  limy_finish_f3[3]),  (x, limx_start_f3[3],  limx_finish_f3[3]))+
                    sp.integrate(f3_xy[4],  (y, limy_start_f3[4],  limy_finish_f3[4]),  (x, limx_start_f3[4],  limx_finish_f3[4]))+
                    sp.integrate(f3_xy[5],  (y, limy_start_f3[5],  limy_finish_f3[5]),  (x, limx_start_f3[5],  limx_finish_f3[5]))+
                    sp.integrate(f3_xy[6],  (y, limy_start_f3[6],  limy_finish_f3[6]),  (x, limx_start_f3[6],  limx_finish_f3[6]))+
                    sp.integrate(f3_xy[7],  (y, limy_start_f3[7],  limy_finish_f3[7]),  (x, limx_start_f3[7],  limx_finish_f3[7]))+
                    sp.integrate(f3_xy[8],  (y, limy_start_f3[8],  limy_finish_f3[8]),  (x, limx_start_f3[8],  limx_finish_f3[8]))+
                    sp.integrate(f3_xy[9],  (y, limy_start_f3[9],  limy_finish_f3[9]),  (x, limx_start_f3[9],  limx_finish_f3[9]))+
                    sp.integrate(f3_xy[10], (y, limy_start_f3[10], limy_finish_f3[10]), (x, limx_start_f3[10], limx_finish_f3[10])
                    ))
    fbar_3 = sp.simplify(fbar_3)
    with open('./test_files/final_variance_equation_3.txt','wb') as f:
                for idx in tqdm(range(len(f3_xy))):
                        variance_f3 = sp.simplify(sp.integrate(((f3_xy[idx]-fbar_3)**2), (y, limy_start_f3[idx],  limy_finish_f3[idx]),  (x, limx_start_f3[idx],  limx_finish_f3[idx])))
                        pickle.dump(str(variance_f3),f)

############# f_4(x,y) ####################
def fun4():
    f4_xy = [(rho*t/vx),
            ((rho*x)/vx),
            ((rho*((-x+w+vx*t))/(vx))),
            (rho*((x)/(vx))),
            ((rho*(-y+h+vy*t))/(vx*vy)),
            (rho*((-x+w+vx*t)/(vx))),
            ((rho*(y)/(vy*vx))),
            ((rho*(y))/(vy*vx)),
            ((rho*(-y+h+vy*t))/(vx*vy)),
            ((rho*((x-w)/(vx*w))+(y/(vx*vy)))),
            ((rho*(-((t*(x-w-vx*t))/(vx*w))-((y-h)/(vx*vy)))))]
            

    limx_start_f4  = [w,0,vx*t,0,0,vx*t,vx*t,w,w,0,vx*t]
    limx_finish_f4 = [vx*t,w,w+vx*t,w,w,w+vx*t,w+vx*t,vx*t,vx*t,w,w+vx*t]
    limy_start_f4  = [vy*t,vy*t,vy*t,h,h+vy*t-((vy*x)/w),((-vy*t*x)/w)+(((vy*t)/w)*(w+vx*t)),0,((-vy*x)/vx)+(vy*t),h,((-vy*x)/vx)+(vy*t),h,h]
    limy_finish_f4 = [h,h,h,h+vy*t-((vy*x)/w),h+vy*t,vy*t,((-vy*t*x)/w)+(((vy*t)/w)*(w+vx*t)),vy*t,(((-vy*x)/vx)+(vy*(w+vx*t)/vx)+w),vy*t,(((-vy*x)/vx)+(vy*(w+vx*t)/vx)+w)]

    fbar_4 =   (sp.integrate(f4_xy[0],  (y, limy_start_f4[0],  limy_finish_f4[0]),  (x, limx_start_f4[0],  limx_finish_f4[0]))+
                sp.integrate(f4_xy[1],  (y, limy_start_f4[1],  limy_finish_f4[1]),  (x, limx_start_f4[1],  limx_finish_f4[1]))+
                sp.integrate(f4_xy[2],  (y, limy_start_f4[2],  limy_finish_f4[2]),  (x, limx_start_f4[2],  limx_finish_f4[2]))+
                sp.integrate(f4_xy[3],  (y, limy_start_f4[3],  limy_finish_f4[3]),  (x, limx_start_f4[3],  limx_finish_f4[3]))+
                sp.integrate(f4_xy[4],  (y, limy_start_f4[4],  limy_finish_f4[4]),  (x, limx_start_f4[4],  limx_finish_f4[4]))+
                sp.integrate(f4_xy[5],  (y, limy_start_f4[5],  limy_finish_f4[5]),  (x, limx_start_f4[5],  limx_finish_f4[5]))+
                sp.integrate(f4_xy[6],  (y, limy_start_f4[6],  limy_finish_f4[6]),  (x, limx_start_f4[6],  limx_finish_f4[6]))+
                sp.integrate(f4_xy[7],  (y, limy_start_f4[7],  limy_finish_f4[7]),  (x, limx_start_f4[7],  limx_finish_f4[7]))+
                sp.integrate(f4_xy[8],  (y, limy_start_f4[8],  limy_finish_f4[8]),  (x, limx_start_f4[8],  limx_finish_f4[8]))+
                sp.integrate(f4_xy[9],  (y, limy_start_f4[9],  limy_finish_f4[9]),  (x, limx_start_f4[9],  limx_finish_f4[9]))+
                sp.integrate(f4_xy[10], (y, limy_start_f4[10], limy_finish_f4[10]), (x, limx_start_f4[10], limx_finish_f4[10])
                ))/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))
    fbar_4 = sp.simplify(fbar_4)
    with open('./test_files/final_variance_equation_4.txt','wb') as f:
                for idx in tqdm(range(len(f4_xy))):
                        variance_f4 = sp.simplify(sp.integrate(((f4_xy[idx]-fbar_4)**2), (y, limy_start_f4[idx],  limy_finish_f4[idx]),  (x, limx_start_f4[idx],  limx_finish_f4[idx])))
                        pickle.dump(str(variance_f4),f)

############# f_5(x,y) ####################
def fun5():
#     f5_xy = [(rho*t/vy),
#             ((rho*x)/(vx*vy)),
#             ((rho*((-x+w+vx*t))/(vx*vy))),
#             (rho*(((y))/(vy))),
#             (rho*((-y+h+vy*t)/(vy))),
#             ((rho*((t*y/(vy*h))-((x-w)/(vx*vy))))),
#             ((rho*(((x/(vx*vy))-(t/(vy*h))*(y-vy*t))))),
#             ((rho*(x))/(vx*vy)),
#             (rho*(((y))/(vy))),
#             (rho*(((-x+w+vx*t))/(vx*vy))),
#             (rho*((-y+h+vy*t)/(vy)))]

    f5_xy = [(rho*t/vx),
            ((rho*x)/(vx)),
            ((rho*((-x+w+vx*t))/(vx))),
            (rho*(((y))/(vy*vx))),
            (rho*((-y+h+vy*t)/(vy*vx))),
            ((rho*((t*y/(h))-((x-w)/(vy))))),
            ((rho*(((x/(vx))-(t/(vy*h))*(y-vy*t))))),
            ((rho*(x))/(vx)),
            (rho*(((y))/(vy*vx))),
            (rho*(((-x+w+vx*t))/(vx))),
            (rho*((-y+h+vy*t)/(vy*vx)))]

    limx_start_f5  = [vx*t,0,w,vx*t,vx*t,w,0,0,0,w,w]
    limx_finish_f5 = [w,vx*t,w+vx*t,w,w,w+vx*t,vx*t,vx*t,vx*t,w+vx*t,w+vx*t]
    limy_start_f5  = [h,h,(vy/vx)*(x-w),0,vy*t,(vy/vx)*(x-w),vy*t,((h*x)/vx*t),0,vy*t,(h*x)/(vx*t)+vy*t-(w*h)/(vx*t)]
    limy_finish_f5 = [vy*t, ((vy/vx)*x)+h,vy*t,h,h+vy*t,h,((vy/vx)*x)+h,h,((h*x)/(vx*t)),(h*x)/(vx*t)+vy*t-(w*h)/(vx*t),h+vy*t]

    fbar_5 =   (sp.integrate(f5_xy[0],  (y, limy_start_f5[0],  limy_finish_f5[0]),  (x, limx_start_f5[0],  limx_finish_f5[0]))+
                sp.integrate(f5_xy[1],  (y, limy_start_f5[1],  limy_finish_f5[1]),  (x, limx_start_f5[1],  limx_finish_f5[1]))+
                sp.integrate(f5_xy[2],  (y, limy_start_f5[2],  limy_finish_f5[2]),  (x, limx_start_f5[2],  limx_finish_f5[2]))+
                sp.integrate(f5_xy[3],  (y, limy_start_f5[3],  limy_finish_f5[3]),  (x, limx_start_f5[3],  limx_finish_f5[3]))+
                sp.integrate(f5_xy[4],  (y, limy_start_f5[4],  limy_finish_f5[4]),  (x, limx_start_f5[4],  limx_finish_f5[4]))+
                sp.integrate(f5_xy[5],  (y, limy_start_f5[5],  limy_finish_f5[5]),  (x, limx_start_f5[5],  limx_finish_f5[5]))+
                sp.integrate(f5_xy[6],  (y, limy_start_f5[6],  limy_finish_f5[6]),  (x, limx_start_f5[6],  limx_finish_f5[6]))+
                sp.integrate(f5_xy[7],  (y, limy_start_f5[7],  limy_finish_f5[7]),  (x, limx_start_f5[7],  limx_finish_f5[7]))+
                sp.integrate(f5_xy[8],  (y, limy_start_f5[8],  limy_finish_f5[8]),  (x, limx_start_f5[8],  limx_finish_f5[8]))+
                sp.integrate(f5_xy[9],  (y, limy_start_f5[9],  limy_finish_f5[9]),  (x, limx_start_f5[9],  limx_finish_f5[9]))+
                sp.integrate(f5_xy[10], (y, limy_start_f5[10], limy_finish_f5[10]), (x, limx_start_f5[10], limx_finish_f5[10])
                ))/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))
    fbar_5 = sp.simplify(fbar_5)
    with open('./test_files/final_variance_equation_5_v3.txt','wb') as f:
                for idx in tqdm(range(len(f5_xy))):
                        variance_f5 = sp.simplify(sp.integrate(((f5_xy[idx]-fbar_5)**2), (y, limy_start_f5[idx],  limy_finish_f5[idx]),  (x, limx_start_f5[idx],  limx_finish_f5[idx])))
                        pickle.dump(str(variance_f5),f)

############# f_6(x,y) ####################
def fun6():
    f6_xy = [(rho*t/vy),
            ((rho*x)/(vx*vy)),
            ((rho*((-x+w+vx*t))/(vx*vy))),
            (rho*(((y))/(vy))),
            (rho*((-y+h+vy*t)/(vy))),
            (rho*(((y))/(vy))),
            (rho*(((-x+w+vx*t))/(vx*vy))),
            (rho*((-y+h+vy*t)/(vy))),
            ((rho*(x))/(vx*vy)),
            ((rho*((x-vx*t)/(vx*vy))+((t*y)/(vy*h)))),
            (rho*((w-x)/(vx*vy))-((t*(y-h-vy*t))/(vy*h)))]

    limx_start_f6  = [vx*t,0,w,vx*t,vx*t,w,w,0,0,0,w]
    limx_finish_f6 = [w,vx*t,w+vx*t,w,w,w+vx*t, w+vx*t,vx*t,vx*t,vx*t,w+vx*t]
    limy_start_f6  = [h,((-vy*x)/vx)+vy*t,h,0,vy*t,0,((h)/(vx*t))*(-x+w+vx*t),((-h*x)/(vx*t))+h+vy*t,vy*t,0,vy*t]
    limy_finish_f6 = [vy*t,vy*t,vy*t,h,h+vy*t,((h)/(vx*t))*(-x+w+vx*t),h,h+vy*t,((-h*x)/(vx*t))+h+vy*t,h,((-vy/vx)*x)+((vy/vx)*w)+h+vy*t]

    fbar_6 =   (sp.simplify(sp.integrate(f6_xy[0],  (y, limy_start_f6[0],  limy_finish_f6[0]),  (x, limx_start_f6[0],  limx_finish_f6[0])))+
                sp.simplify(sp.integrate(f6_xy[1],  (y, limy_start_f6[1],  limy_finish_f6[1]),  (x, limx_start_f6[1],  limx_finish_f6[1])))+
                sp.simplify(sp.integrate(f6_xy[2],  (y, limy_start_f6[2],  limy_finish_f6[2]),  (x, limx_start_f6[2],  limx_finish_f6[2])))+
                sp.simplify(sp.integrate(f6_xy[3],  (y, limy_start_f6[3],  limy_finish_f6[3]),  (x, limx_start_f6[3],  limx_finish_f6[3])))+
                sp.simplify(sp.integrate(f6_xy[4],  (y, limy_start_f6[4],  limy_finish_f6[4]),  (x, limx_start_f6[4],  limx_finish_f6[4])))+
                sp.simplify(sp.integrate(f6_xy[5],  (y, limy_start_f6[5],  limy_finish_f6[5]),  (x, limx_start_f6[5],  limx_finish_f6[5])))+
                sp.simplify(sp.integrate(f6_xy[6],  (y, limy_start_f6[6],  limy_finish_f6[6]),  (x, limx_start_f6[6],  limx_finish_f6[6])))+
                sp.simplify(sp.integrate(f6_xy[7],  (y, limy_start_f6[7],  limy_finish_f6[7]),  (x, limx_start_f6[7],  limx_finish_f6[7])))+
                sp.simplify(sp.integrate(f6_xy[8],  (y, limy_start_f6[8],  limy_finish_f6[8]),  (x, limx_start_f6[8],  limx_finish_f6[8])))+
                sp.simplify(sp.integrate(f6_xy[9],  (y, limy_start_f6[9],  limy_finish_f6[9]),  (x, limx_start_f6[9],  limx_finish_f6[9])))+
                sp.simplify(sp.integrate(f6_xy[10], (y, limy_start_f6[10], limy_finish_f6[10]), (x, limx_start_f6[10], limx_finish_f6[10]))
                ))/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))
    fbar_6 = sp.simplify(fbar_6)
    with open('./test_files/final_variance_equation_6.txt','wb') as f:
                for idx in tqdm(range(len(f6_xy))):
                        variance_f6 = sp.simplify(sp.integrate(((f6_xy[idx]-fbar_6)**2), (y, limy_start_f6[idx],  limy_finish_f6[idx]),  (x, limx_start_f6[idx],  limx_finish_f6[idx])))
                        print(variance_f6)
                        pickle.dump(str(variance_f6),f)

############# f_7(x,y) ####################
def fun7():
    f7_xy = [(rho*t/(vx*vy)),
            ((rho*x)/(vx*vy)),
            ((rho*((-x+w+vx*t))/(vx*vy))),
            (rho*(((y))/(vx*vy))),
            (rho*((-y+h+vy*t)/(vy*vx))),
            (rho*(((y))/(vy*vx))),
            (rho*(((-x+w+vx*t))/(vx*vy))),
            (rho*((-y+h+vy*t)/(vy*vx))),
            ((rho*((t*x/w)-(t*(y-vy*t)/h))/(vx*vy))),
            (rho*(((-t*(x-vx*t))/(vx*vy*w))+(t*y/(vx*vy*h))))]

    limx_start_f7  = [w,0,vx*t,w,w,0,vx*t,vx*t,0,vx*t]
    limx_finish_f7 = [vx*t,w,w+vx*t,vx*t,vx*t,w,w+vx*t,w+vx*t,w,w+vx*t]
    limy_start_f7  = [(vy/vx)*(x-w),0,(vy/vx)*(x-w),(vy/vx)*(x-w),vy*t,0,vy*t,(((h*x)/w)-((h*vx*t)/w))+vy*t,vy*t,(vy/vx)*(x-w)]
    limy_finish_f7 = [(vy/vx)*x+h,((vy/vx)*x)+h,vy*t,h,(vy/vx)*x+h,((h*x)/(w)),(((h*x)/w)-((h*vx*t)/w))+vy*t,h+vy*t,((vy/vx)*x)+h,h]

    fbar_7 =   (sp.integrate(f7_xy[0],  (y, limy_start_f7[0],  limy_finish_f7[0]),  (x, limx_start_f7[0],  limx_finish_f7[0]))+
                sp.integrate(f7_xy[1],  (y, limy_start_f7[1],  limy_finish_f7[1]),  (x, limx_start_f7[1],  limx_finish_f7[1]))+
                sp.integrate(f7_xy[2],  (y, limy_start_f7[2],  limy_finish_f7[2]),  (x, limx_start_f7[2],  limx_finish_f7[2]))+
                sp.integrate(f7_xy[3],  (y, limy_start_f7[3],  limy_finish_f7[3]),  (x, limx_start_f7[3],  limx_finish_f7[3]))+
                sp.integrate(f7_xy[4],  (y, limy_start_f7[4],  limy_finish_f7[4]),  (x, limx_start_f7[4],  limx_finish_f7[4]))+
                sp.integrate(f7_xy[5],  (y, limy_start_f7[5],  limy_finish_f7[5]),  (x, limx_start_f7[5],  limx_finish_f7[5]))+
                sp.integrate(f7_xy[6],  (y, limy_start_f7[6],  limy_finish_f7[6]),  (x, limx_start_f7[6],  limx_finish_f7[6]))+
                sp.integrate(f7_xy[7],  (y, limy_start_f7[7],  limy_finish_f7[7]),  (x, limx_start_f7[7],  limx_finish_f7[7]))+
                sp.integrate(f7_xy[8],  (y, limy_start_f7[8],  limy_finish_f7[8]),  (x, limx_start_f7[8],  limx_finish_f7[8]))+
                sp.integrate(f7_xy[9],  (y, limy_start_f7[9],  limy_finish_f7[9]),  (x, limx_start_f7[9],  limx_finish_f7[9]))
                )/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))
    fbar_7 = sp.simplify(fbar_7)
    with open('./test_files/final_variance_equation_7.txt','wb') as f:
                for idx in tqdm(range(len(f7_xy))):
                        variance_f7 = sp.simplify(sp.integrate(((f7_xy[idx]-fbar_7)**2), (y, limy_start_f7[idx],  limy_finish_f7[idx]),  (x, limx_start_f7[idx],  limx_finish_f7[idx])))
                        pickle.dump(str(variance_f7),f)

############# f_7(x,y) ####################
def fun8():
    f8_xy = [(rho*t/(vx*vy)),
            ((rho*x)/(vx*vy)),
            ((rho*((y))/(vx*vy))),
            (rho*(((-x+w+vx*t))/(vx*vy))),
            (rho*((-y+h+vy*t)/(vy*vx))),
            (rho*(((-y+h+vy*t))/(vy*vx))),
            (rho*((x)/(vy*vx))),
            ((rho*(-t*(x-vx*t)/(vx*vy*w))-(t*(y-h-vy*t)/(vx*vy*h)))),
            (rho*((t*x/(vx*vy*w))+(t*(y-h)/(vx*vy*h)))),
            (rho*((y)/(vy*vx))),
            (rho*(((-x+w+vx*t))/(vx*vy)))]

    limx_start_f8  = [w,0,w,vx*t,w,0,0,vx*t,0,vx*t,vx*t]
    limx_finish_f8 = [vx*t,w,vx*t,w+vx*t,vx*t,w,w,w+vx*t,w,w+vx*t,w+vx*t]
    limy_start_f8  = [((-vy*x)/vx)+vy*t,((-vy*x)/vx)+vy*t,((-vy*x)/vx)+vy*t,h,vy*t,((-h*x)/(w))+h+vy*t,vy*t,vy*t,((-vy*x)/vx)+vy*t,0,((-h*x)/w)+((h*(w+vx*t))/w)]
    limy_finish_f8 = [(((-vy/vx)*x)+(vy/vx)*w)+h+vy*t,vy*t,h,(((-vy/vx)*x)+(vy/vx)*w)+h+vy*t,(((-vy/vx)*x)+(vy/vx)*w)+h+vy*t,h+vy*t,((-h*x)/(w))+h+vy*t,(((-vy/vx)*x)+(vy/vx)*w)+h+vy*t,h,((-h*x)/w)+((h*(w+vx*t))/w),h]
    
    fbar_8 =   (sp.integrate(f8_xy[0],  (y, limy_start_f8[0],  limy_finish_f8[0]),  (x, limx_start_f8[0],  limx_finish_f8[0]))+
                sp.integrate(f8_xy[1],  (y, limy_start_f8[1],  limy_finish_f8[1]),  (x, limx_start_f8[1],  limx_finish_f8[1]))+
                sp.integrate(f8_xy[2],  (y, limy_start_f8[2],  limy_finish_f8[2]),  (x, limx_start_f8[2],  limx_finish_f8[2]))+
                sp.integrate(f8_xy[3],  (y, limy_start_f8[3],  limy_finish_f8[3]),  (x, limx_start_f8[3],  limx_finish_f8[3]))+
                sp.integrate(f8_xy[4],  (y, limy_start_f8[4],  limy_finish_f8[4]),  (x, limx_start_f8[4],  limx_finish_f8[4]))+
                sp.integrate(f8_xy[5],  (y, limy_start_f8[5],  limy_finish_f8[5]),  (x, limx_start_f8[5],  limx_finish_f8[5]))+
                sp.integrate(f8_xy[6],  (y, limy_start_f8[6],  limy_finish_f8[6]),  (x, limx_start_f8[6],  limx_finish_f8[6]))+
                sp.integrate(f8_xy[7],  (y, limy_start_f8[7],  limy_finish_f8[7]),  (x, limx_start_f8[7],  limx_finish_f8[7]))+
                sp.integrate(f8_xy[8],  (y, limy_start_f8[8],  limy_finish_f8[8]),  (x, limx_start_f8[8],  limx_finish_f8[8]))+
                sp.integrate(f8_xy[9],  (y, limy_start_f8[9],  limy_finish_f8[9]),  (x, limx_start_f8[9],  limx_finish_f8[9]))+
                sp.integrate(f8_xy[10], (y, limy_start_f8[10], limy_finish_f8[10]), (x, limx_start_f8[10], limx_finish_f8[10])
                ))/((w+vx*t)*(h+vy*t)-(vx*vy*t**2))
    fbar_8 = sp.simplify(fbar_8)
    with open('./test_files/final_variance_equation_8.txt','wb') as f:
                for idx in tqdm(range(len(f8_xy))):
                        variance_f8 = sp.simplify(sp.integrate(((f8_xy[idx]-fbar_8)**2), (y, limy_start_f8[idx],  limy_finish_f8[idx]),  (x, limx_start_f8[idx],  limx_finish_f8[idx])))
                        pickle.dump(str(variance_f8),f)

# output = fun1()
# output = fun2()
# output = fun3()
# output = fun4()
output = fun5()
# output = fun6()
#output = fun7()
#output = fun8()

# w           = 1
# h           = 1
# rho         = 1
# t           = 1
# cx          = 0
# cy          = 0
# RESOLUTION  = 200
# VelArray    = np.linspace(0,h*2,RESOLUTION)
# variance_2D = np.empty((RESOLUTION,RESOLUTION))

# for iVely in range(RESOLUTION):
#     cy+=1
#     fieldy = VelArray[iVely]
#     for iVelx in range(RESOLUTION):
#         cx+=1
#         fieldx = VelArray[iVelx]
#         if fieldx >= 0 and fieldy >= 0 and abs(fieldx)<=w and abs(fieldy)<=h or fieldx <= 0 and fieldy <= 0 and abs(fieldx)<=w and abs(fieldy)<=h:
#             vx=abs(fieldx)
#             vy=abs(fieldy)

#             parameters = [
#             (rho, rho),
#             (vx, vx),
#             (vy, vy),
#             (w, w), 
#             (h, h), 
#             (t, t)
#             ]
#             simvar = sp.simplify(variance_f1)

#             variance_2D[cx,cy]=simvar
#             colormap = plt.pyplot.get_cmap("magma")
#             gamma = lambda image: image ** (1 / 2)
#             scaled_pixels = gamma((variance_2D - variance_2D.min()) / (variance_2D.max() - variance_2D.min()))
#             image = PIL.Image.fromarray(
#                 (colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8)
#             )
#             image.show()
        
