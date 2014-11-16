#ifndef _NN_SLP_H
#define _NN_SLP_H

#include <vector>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif


#define DTYPE float

using namespace std;


class NN_MLP_SingleHiddenLayer {
	private:

		uint _epochs;
		uint _bias;
		uint _cInp;
		uint _ci, _ch, _co;
// 		DTYPE _lRate, _mRate;

		cl_context _ctx;
		cl_command_queue _queue;
		cl_int _cDevice;
		vector<cl_device_id> _devices;
		cl_kernel _ker_activate, _ker_backPropagation;
		cl_int _ert;

		DTYPE *_ai, *_ah, *_ao;
		DTYPE *_wi, *_wo;
		DTYPE *_pi, *_po; // previous delta
		DTYPE *_th, *_to; // threshold
// 		DTYPE *_errs;

		cl_mem _ai_b, _ah_b, _ao_b;
		cl_mem _wi_b, _wo_b;
		cl_mem _pi_b, _po_b;
		cl_mem _th_b, _to_b;
		cl_mem _tars_b;
    uint cg;
	// 		_errs_b;po

		void bindBuffers();

	public:

		uint cDataset = 0;
		DTYPE *dataset_i, *dataset_o;
		size_t *gSize = nullptr;
		size_t *lSize = nullptr;

		NN_MLP_SingleHiddenLayer(uint ci, uint ch, uint co, uint bias,
			   DTYPE wRange = .3, DTYPE tRange = 0.);

		~NN_MLP_SingleHiddenLayer();

		void creatContext(
			cl_device_type tarDevType = CL_DEVICE_TYPE_GPU, int tarDevId = -1);

		cl_kernel loadKernel(string fName, string kerName, string flags = "");

		void setBuffers();
		DTYPE *getBuffers(bool print=false);

		void setupCL(
			string fName, string flags = "",
			cl_device_type tarDevType = CL_DEVICE_TYPE_GPU, int tarDevId = -1);

		uint getEpochs();
		void debugW();
		void debugT();
		void debugA();

		void activate(DTYPE *inps);
		DTYPE backPropagation(DTYPE *tars);

		DTYPE train(uint epochs, DTYPE lRate, DTYPE mRate);

		DTYPE normalize(DTYPE x);
		DTYPE denormalize(DTYPE x);


};

#endif
