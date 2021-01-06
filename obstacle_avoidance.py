

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import sqrtm
from scipy import ndimage
import csv

# this script computes the optimum trajectory via gradient descent in an obstacle environment


Nrows=400
Ncols=600

x = np.arange(0, 600, 1)
y = np.arange(0, 400, 1)

xx, yy = np.meshgrid(x, y, sparse=True)
z= np.zeros((Nrows ,Ncols))
z1=z


# obstacle generate 
z [300:, 100:250] = 200
z [100:150, 300:400] = 200

# obstacle 1 generation through a function
obs_circle1_x=200
obs_circle1_y=150
t = ((xx - obs_circle1_x)**2 + (yy - obs_circle1_y)**2) < 50**2
z [t] = 200 

# obstacle 2 generation through a function
obs_circle2_x=400
obs_circle2_y=300
t = ((xx - obs_circle2_x)**2 + (yy - obs_circle2_y)**2) < 100**2
z [t] = 200 

# Compute distance transform
d =ndimage.distance_transform_edt(z)
# Rescale and transform distances
d2 = (d/100) + 1
d0 = 2
nu = 800
repulsive = nu*(((d2**-1) - (d0**-1))**2)
repulsive[d2 > d0] = 0
plot3= plt.figure(1)
ax = plot3.gca(projection='3d')
surf = ax.plot_surface(xx, yy, repulsive, cmap=cm.RdBu,
                       linewidth=0, antialiased=False)



# gradient field towards goal
goal =[400, 50]
xi = 1/700
attractive = xi * ( (xx - goal[0])**2 + (yy - goal[1])**2 )

plot3= plt.figure(2)
ax = plot3.gca(projection='3d')
surf = ax.plot_surface(xx, yy, attractive, cmap=cm.RdBu,
                       linewidth=0, antialiased=False)


total=attractive-repulsive

plot4= plt.figure(3)
ax = plot4.gca(projection='3d')
surf = ax.plot_surface(xx, yy, total, cmap=cm.RdBu,
                       linewidth=0, antialiased=False)
plt.ylabel('Y')
plt.xlabel('X')


gx1,gy1=np.gradient(-total)


A=gx1**2+gy1**2
mag1=np.sqrt(A)
# normalize the gradient
gx = np.true_divide(gy1,mag1)
gy = np.true_divide(gx1,mag1)

## gradient descent algorithm
routeX=[]
routeY=[]
start = [50, 350]
end_coords=goal
coord=start
iter=0
V= 2#5

max_its=1000

while iter < max_its :
    
    dist=math.sqrt((end_coords[0]-coord[0])**2+(end_coords[1]-coord[1])**2)

    if dist < 2 :
            break
    

    grad_x=(gx[coord[1],coord[0]])
    grad_y=(gy[coord[1],coord[0]])

    posx=coord[0]+V*grad_x
    posy=coord[1]+V*grad_y
  
    routeX.append(posx)
    routeY.append(posy)

    coord=[int(posx),int(posy)]
    iter=iter+1


routeX[:0] = [start[0]]
routeY[:0] = [start[1]]




plot1 = plt.figure(4)
plt.contourf(x,y,z)
plt.plot(routeX,routeY)
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


