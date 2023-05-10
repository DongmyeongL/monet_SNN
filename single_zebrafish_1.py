import snn_dm_python as snn
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

print("setting.....");
r_cell_xyz=np.load('./data/zebrafish_cell_xyz.npy');
r_sc=np.load('./data/zebrafish_sc.npy')

plt.scatter(r_cell_xyz[:,0],r_cell_xyz[:,1],r_cell_xyz[:,2]);
plt.show();
c=1


nn=len(r_cell_xyz);
print(nn)

area_index=[];
for i in range(72):
    area_index.extend(np.where(r_cell_xyz[:,3]==(i+1)));

group_area=[];

for k in range(72):
    xx=[];
    for i in range(nn):
         if r_cell_xyz[i][3]==k:
               xx.append(i);
    group_area.append(xx);

    
inhibition_data=[];
for i in range(nn):
    if random.random()<0.3 :
        inhibition_data.append(i);



connect_data=[];


qq=[];


t_num=0;


init_xx=[];

init_xx.append(12);

for i in range(36):
    init_xx.append(10);


for i in range(72):
    for j in range(i+1,72):
           for k in range(int(r_sc[i,j]*2)):

                l1=len(area_index[i]);
                l2=len(area_index[j]);

                if(l1>0 and l2>0):
                    s1=random.randrange(0,l1);
                    s2=random.randrange(0,l2);

                    if(random.random()>0.5):
                       connect_data.append([area_index[i][s1],area_index[j][s2], init_xx[0],0]);
                    else:
                       connect_data.append([area_index[j][s2],area_index[i][s1],init_xx[0],0]);

for k in range(72):

    xx=[];

    for i in range(nn):
        if r_cell_xyz[i][3]==k:
            xx.append(i)


    for i in range(len(xx)):
        for j in range(i+1,len(xx)):
            if(random.random()<0.2):

                    dx=r_cell_xyz[xx[i]][0]-r_cell_xyz[xx[j]][0];
                    dy=r_cell_xyz[xx[i]][1]-r_cell_xyz[xx[j]][1];
                    dz=r_cell_xyz[xx[i]][2]-r_cell_xyz[xx[j]][2];

                    dd=np.sqrt(dx*dx+dy*dy+dz*dz);

                    if(dd<20) :
                        if(random.random()<0.5):
                            connect_data.append([xx[i],xx[j],init_xx[k%36+1],k+1]);
                        else:
                            connect_data.append([xx[j],xx[i],init_xx[k%36+1],k+1]);
                                
                         

print("cuda running.....");
time=15;
dd_old,av,cc=snn.fun_stim_large_scaling(nn,11.0,time,area_index,inhibition_data, connect_data);
   


for i in range(72):
  plt.plot(av[i,:]-i*0.04);

plt.show();

plt.imshow(cc)
plt.show()
