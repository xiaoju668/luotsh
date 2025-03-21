import numpy as np
import matplotlib.pyplot as plt

def change_xv(x,p):
    k11 = 0.126 * p
    k21 =  -(0.126 * x)
    k12 = (p + 0.5*0.2*k21)*0.126
    k22 = -(x + 0.5*0.2*k11*0.126)*0.126
    k13 = (p + 0.5*0.2*(k22))*0.126
    k23 = -(x + 0.5*0.2*k12*0.126)*0.126
    k14 = (p + 0.2*(k23))*0.126
    k24 = -(x + 0.2*k13*0.126)*0.126
    next_x = x + (1/6)*0.2*(k11 + 2*k12 + 2*k13 + k14)
    next_p = p + (1/6)*0.2*(k21 + 2*k22 + 2*k23 + k24)
    return next_x,next_p

x = 0.7071
p = 0.7071
x_s = []
p_s = []
for i in range(500):
    next_x,next_p = change_xv(x,p)
    x = next_x
    p = next_p
    x_s.append(x)
    p_s.append(p)
times_draw = np.arange(0,500)
plt.plot(times_draw,p_s,color='red',linestyle='--')
plt.plot(times_draw,x_s)
plt.show()