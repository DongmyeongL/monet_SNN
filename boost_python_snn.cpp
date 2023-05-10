#include <boost/python.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include "CUDA_DM_NEURON.cuh"
#include "CUDA_DM_RUN.cuh""
#include "opengl_dis.h"

template<typename T> inline
std::vector< T > py_list_to_std_vector(const boost::python::object& iterable)
{
    return std::vector< T >(boost::python::stl_input_iterator< T >(iterable),
        boost::python::stl_input_iterator< T >());
};


template <class T> inline
boost::python::list std_vector_to_py_list(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
};


struct boost_neuronal_netowrk :s_neuronal_netowrk
{

    void set_stimulus(boost::python::list& neuron_num, boost::python::list& stimulus)
    {
        auto ll = boost::python::len(neuron_num);
        this->stimulus_neuron_number.clear();
        this->stimulus_data.clear();

        for (int i = 0; i < ll; i++)
        {
            this->stimulus_neuron_number.push_back(boost::python::extract<int>(neuron_num[i]));
        }

        this->stimulus_data.resize(ll);

        for (int i = 0; i < ll; i++)
        {
            boost::python::list _data = boost::python::extract<boost::python::list>(stimulus[i]);

            auto pp = boost::python::len(_data);


            for (int j = 0; j < pp; j++)
            {
                this->stimulus_data[i].push_back(boost::python::extract<double>(_data[j]));
            }
        }

        this->stim_number = this->stimulus_neuron_number.size();

        if (this->stimulus_data.size() > 0)
        {
            this->stim_length = this->stimulus_data[0].size();
        }
        else
        {
            this->stim_length = 0;
        }

        std::vector<double> t_data;
        for (int i = 0; i < stimulus_data.size(); i++)
            for (int j = 0; j < stimulus_data[i].size(); j++)
                t_data.push_back(stimulus_data[i][j]);


        cudaFree(p_stim_number);
        cudaFree(p_stim_data);


        cudaMalloc((void**)&p_stim_number, stim_number * sizeof(int));
        cudaMalloc((void**)&p_stim_data, stim_length * stim_number * sizeof(double));


        cudaMemcpy(this->p_stim_number, (void*)&this->stimulus_neuron_number[0], stim_number * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(this->p_stim_data, (void*)&t_data[0], stim_length * stim_number * sizeof(double), cudaMemcpyHostToDevice);

        
    }
     
    void run_display();

       
    boost::python::list  get_spike_data()
    {
        
        boost::python::list _out_data;
        if (this->run_param.spiking_time_record == true)
        {
            auto ll = (this->out_data.size());
            
            for (int i = 0; i < ll; i++)
            {
                boost::python::list t_out_data;
            
                t_out_data.append(this->out_data[i].x);
                t_out_data.append(this->out_data[i].y);
                _out_data.append(t_out_data);
            }

            
        }

        return _out_data;
    }

    boost::python::list  get_ca_data() 
    {
        boost::python::list ca_out_data;
        
        if (this->calcium_recording == true)
        {
            for (int i = 0; i < this->ca_ll; i++)
            {
                boost::python::list t_out_data;

                for (int j = 0; j< this->nn_neuron_num; j++)
                {
                    t_out_data.append(this->out_ca_data[i * this->nn_neuron_num + j]);
                }

                ca_out_data.append(t_out_data);

            }
        }

        return ca_out_data;
    }

    void  cuda_run_python()
    {
        //std::vector<s_xy_data> out_data;
        this->cuda_run_stdp();
    }
    
    boost::python::list get_weight_matrix()
    {
        boost::python::list _out_data;

        for (int i = 0; i < this->nn_neuron_num; i++)
        {
            boost::python::list t_out_data;


            std::vector<double> tt(this->nn_neuron_num,0.0);
            
                        

            for (int j = 0; j < this->_connect_data[i].s_pre_id.size(); j++)
            {
                int ii = this->_connect_data[i].s_pre_id[j];


             
               tt[ii] = this->_connect_data[i].weight[j];

            }

            for (int j = 0; j < this->nn_neuron_num; j++)
            {
                t_out_data.append(tt[j]);
            }

            _out_data.append(t_out_data);
        }


        return _out_data;

    }

};


char const* greet()
{
    return "Monet_SNN_DM, world wow";
}

using namespace boost::python;

BOOST_PYTHON_MODULE(MONET_SNN_CUDA_PYTHON)
{
    def("greet", greet);


    class_<boost_neuronal_netowrk>("DM_SNN")
        .def("all_clear", &boost_neuronal_netowrk::all_clear)
        .def("set_neuron_number", &boost_neuronal_netowrk::set_neuron_number)
        .def("get_neuron_number", &boost_neuronal_netowrk::get_neuron_number)
        .def("set_inhibtion_neuron", &boost_neuronal_netowrk::set_inhibtion_neuron)
        .def("get_inhibtion_neuron", &boost_neuronal_netowrk::get_inhibtion_neuron)
        .def("set_connection", &boost_neuronal_netowrk::set_connection)
        .def("set_neuron_xyz", &boost_neuronal_netowrk::set_neuron_xyz)
        .def("set_stdp_param", &boost_neuronal_netowrk::set_stdp_param)
        .def("set_run_param", &boost_neuronal_netowrk::set_run_param)
        .def("set_stimulus", &boost_neuronal_netowrk::set_stimulus)
        .def("get_version", &boost_neuronal_netowrk::get_version)
        .def("get_spike_data", &boost_neuronal_netowrk::get_spike_data)
        .def("get_ca_data", &boost_neuronal_netowrk::get_ca_data)
        .def("cuda_run_python", &boost_neuronal_netowrk::cuda_run_python)
        .def("create_cuda_memory", &boost_neuronal_netowrk::create_cuda_memory)
        .def("run_display", &boost_neuronal_netowrk::run_display)
        .def("set_calcium_recording", &boost_neuronal_netowrk::set_calcium_recording)
        ;

}





void  boost_neuronal_netowrk::run_display()
{
    int argc = 0;
    char** argv;

    //set_intensity(noise);
    this->run_param.spiking_time_record = false;
    this->calcium_recording = false;

    srand(GetCurrentProcessId());
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

    int x = 20;
    int y = 10;
    glutInitWindowPosition(x, y);
    int win = glutCreateWindow("MONET SNN CORTICAL COLUM MODEL");
    printf("window id: %d\n", win);

    InitializeGlutCallbacks_1();

    // Must be done after glut is initialized!
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return;
    }

    GLclampf Red = 0.0f, Green = 0.0f, Blue = 0.0f, Alpha = 0.0f;
    glClearColor(Red, Green, Blue, Alpha);

    set_neural_network(this);
    Create_neuronal_network2();
 
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glCullFace(GL_BACK);

    CompileShaders();

    glutMainLoop();

}



