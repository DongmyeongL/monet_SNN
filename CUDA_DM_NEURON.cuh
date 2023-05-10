#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>


struct s_stdp_param 
{
	double w_max = 8;
	double p_rate = 0.2;
	double n_rate = 0.2;

};
struct s_neuron_connection
{
	std::vector<double> weight;
	std::vector<unsigned int> s_pre_id;
};

struct cuda_s_izkevich
{
	double v = -70, u = 0;
	double c1 = 0.04, c2 = 5, c3 = 140, c4 = 1, c5 = -1, c6 = -1;
	//double c1 = 0.04, c2 = 5, c3 = 140, c4 = 1, c5 =- 3, c6 = 0;

	double E_exc=0.0, E_inh=0.0;

	double a = 0.02, b = 0.2, c = -65, d = 8;//d=8;
	double thre = 50;

	double exc_synap = 0;
	double inh_synap = 0;

	double gamma_exc = 10.0;
	double gamma_inh = 5.0;;

	bool spike_checking = false;
	bool check_inh = false;

	double calcium = 0;
	double spiking_time = -10000.0;

	double external_input = 0;
	//std::vector<int> qq;


	/*
	__device__ double fun_v();
	__device__ double fun_u();
	__device__ void checking_spike(const double st);
	__device__ void exc_synapse_model(const double inter, const double intra, const double dt);
	__device__ void inh_synapse_model(const double intra, const double inter, const double dt);
	*/


	__device__ double fun_v()
	{
		return c1 * v * v + c2 * v + c3 - c4 * u + c5 * exc_synap * (v - 10.0) + c6 * inh_synap * (v + 70);
	}


	__device__ double fun_u()
	{
		return a * (b * v - u);
	};

	__device__ void checking_spike(const double st)
	{
		if (this->v >= this->thre)
		{
			this->v = this->c;
			this->u += this->d;
			this->spike_checking = true;
			this->spiking_time = st;
		}
		else
		{
			this->spike_checking = false;
		}
		return;
	}
	
	__device__ void exc_synapse_model(const double dt)
	{
		double temp = -this->gamma_exc * this->exc_synap + this->E_exc;
		this->exc_synap += dt * temp;

		return;
	}

	__device__ void inh_synapse_model( const double dt)
	{
		double temp = -this->gamma_inh * this->inh_synap + this->E_inh;
		this->inh_synap += dt * temp;

		return;
	}
	
	/*
	void exc_synapse_model(const double inter, const double dt)
	{
		double temp = -this->gamma_exc * this->exc_synap + inter;
		this->exc_synap += dt * temp;

		return;
	}

	 void inh_synapse_model(const double inter, const double dt)
	{
		double temp = -this->gamma_inh * this->inh_synap + inter;
		this->inh_synap += dt * temp;

		return;
	}
	*/
};



