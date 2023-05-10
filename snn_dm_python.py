import MONET_SNN_CUDA_PYTHON as snn
import numpy as np

def xx(xx):
    return xx;

def fun_stim_large_scaling(nn, noise_intensity, time,area_index, inhibition_data, connect_data):

    _neurnal=snn.DM_SNN();
    _neurnal.all_clear();

    _neurnal.set_neuron_number(nn);

    zebra_fc=np.load('./data/zebra_fc.npy');


    for i in inhibition_data:
        _neurnal.set_inhibtion_neuron(i);

    for j in range(len(connect_data)):
        _neurnal.set_connection(int(connect_data[j][0]),int(connect_data[j][1]),float(connect_data[j][2]));


    r_cell_xyz=np.load('./data/zebrafish_cell_xyz.npy');


    for i in range(len(r_cell_xyz)):
        _neurnal.set_neuron_xyz(i,r_cell_xyz[i][0],r_cell_xyz[i][1],r_cell_xyz[i][2]*2,0.9,0.8,0.3);

    
    ll=20*1000*time;
    _neurnal.set_run_param(0.05,0,ll,noise_intensity ,False,False,False);
    

    _neurnal.create_cuda_memory();
    _neurnal.set_calcium_recording(2000);

    
    #_neurnal.cuda_run_python();

    _neurnal.run_display();

    

    spike_data=_neurnal.get_spike_data();
    ca_data=_neurnal.get_ca_data();

    

    ca_data=np.array(ca_data);
    ca_data=ca_data.transpose();
    
    average_data=[];

    pl=len(ca_data[1,:])
    
    for i in range(72):

        if(len(area_index[i])>0):
            xx=ca_data[area_index[i],:];
            average_data.append(np.mean(xx.transpose(),1));
      
        else:
   
            average_data.extend(np.zeros((1,pl)).tolist());
    

    av=np.zeros([72,pl]);

    for i in range(72):
        for j in range(pl):
            av[i,j]=average_data[i][j];

    av=av+np.random.normal(0,0.005,[72,pl]);

  
    cc=np.corrcoef(av)

    da=np.abs(zebra_fc-cc)
    dd=np.sum(da);
    for i in range(72):
        cc[i,i]=0.0; 

    return dd,av,cc;

