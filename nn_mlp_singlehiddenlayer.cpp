
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

#include <unistd.h>
// #include <time.h>
#include <math.h>


#include "nn_mlp_singlehiddenlayer.h"


using namespace std;


#ifdef PERFORMANCE
	#warning PERFORMANCE ON!
	#define checkErr(a,b,c) 0
#else
	inline void checkErr(uint lineNo, const char *reason, cl_int err){
		if (err != CL_SUCCESS) {
			cout << "Error at line " << lineNo <<
			" in " << reason <<
			" with " << err << endl;
			exit(err);
		}
	}
#endif



NN_MLP_SingleHiddenLayer::NN_MLP_SingleHiddenLayer(uint ci, uint ch, uint co, uint bias,
			   DTYPE wRange, DTYPE tRange, int flags){

	_epochs = 0;
	_bias = bias;
	_cInp = ci;
	_ci = ci + bias;
	_ch = ch + bias;
	_co = co;
	_flags = flags;

	_ai = (DTYPE *)malloc(sizeof(DTYPE) * _ci);
	_ah = (DTYPE *)malloc(sizeof(DTYPE) * _ch);
	_ao = (DTYPE *)malloc(sizeof(DTYPE) * _co);

	_wi = (DTYPE *)malloc(sizeof(DTYPE) * _ci * _ch);
	_wo = (DTYPE *)malloc(sizeof(DTYPE) * _ch * _co);

	if (flags & NN_FLAG_MOMENTUM) {
		_pi = (DTYPE *)malloc(sizeof(DTYPE) * _ci * _ch);
		_po = (DTYPE *)malloc(sizeof(DTYPE) * _ch * _co);
	}

	if (flags & NN_FLAG_THRESHOLD) {
		_th = (DTYPE *)malloc(sizeof(DTYPE) * _ch);
		_to = (DTYPE *)malloc(sizeof(DTYPE) * _co);
	}

// 	_errs = (DTYPE *)malloc(sizeof(DTYPE) * _co);

	uint i;
	for (i=ci; i<_ci; i++) _ai[i] = 1.;
	for (i=ch; i<_ch; i++) _ah[i] = 1.;

	srand(time(NULL));

	for (uint i = 0;i<_ci;i++)
		for (uint j=0;j<_ch;j++) {
			_wi[i*_ch + j] = (random()/(DTYPE)(RAND_MAX/2)-1.) *wRange;
			if (flags & NN_FLAG_MOMENTUM) _pi[i*_ch + j] = 0;
		}

	for (uint i = 0;i<_ch;i++)
		for (uint j=0;j<_co;j++) {
			_wo[i*_co + j] = (random()/(DTYPE)(RAND_MAX/2)-1.) *wRange;
			if (flags & NN_FLAG_MOMENTUM) _po[i*_co + j] = 0;
		}

	if (flags & NN_FLAG_THRESHOLD) {
		for (uint i=0;i<_ch;i++) _th[i] = (random()/(DTYPE)(RAND_MAX/2)-1.) *tRange;
		for (uint i=0;i<_co;i++) _to[i] = (random()/(DTYPE)(RAND_MAX/2)-1.) *tRange;
	}

}



void NN_MLP_SingleHiddenLayer::creatContext(
	cl_device_type tarDevType, int tarDevId){

	cl_uint cPlatform = 0, cDevice = 0;
	size_t sName;
	string name;
	uint i;

	_ert = clGetPlatformIDs(0, nullptr, &cPlatform);
	checkErr(__LINE__, "clGetPlatformIDs", _ert);

	cout << "<CreateContext>" << endl;
	cout << "Platforms:" << cPlatform << endl;

	vector<cl_platform_id> platforms(cPlatform);
	_ert = clGetPlatformIDs(cPlatform, platforms.data(), nullptr);
	checkErr(__LINE__, "clGetPlatformIDs", _ert);

	for (i=0;i<cPlatform;i++){
		_ert = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &sName);
		name.resize(sName);
		_ert = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sName, const_cast<char*> (name.data()), nullptr);

		cout << '\t' << name << endl;
	}

	if (tarDevId >= 0) {
		_ert = clGetDeviceIDs(platforms[tarDevId], tarDevType,
							  0, nullptr, &cDevice);
	}
	else {
		for (i=0;i<cPlatform;i++){
			_ert = clGetDeviceIDs(platforms[i], tarDevType,
				0, nullptr, &cDevice);
			if (cDevice > 0) break;
		}
		tarDevId = i;
	}
	checkErr(__LINE__, "clGetDeviceIDs", _ert);

	cout << "Select:" << tarDevId << endl;

	vector<cl_device_id> devices(cDevice);
	_ert = clGetDeviceIDs(platforms[tarDevId], tarDevType,
						  cDevice, devices.data(), nullptr);
	checkErr(__LINE__, "clGetDeviceIDs", _ert);

	_ert = clGetDeviceInfo(devices[0], CL_DEVICE_NAME,
						   0, nullptr, &sName);
	name.resize(sName);
	_ert = clGetDeviceInfo(devices[0], CL_DEVICE_NAME,
						   sName, const_cast<char*> (name.data()), nullptr);
	checkErr(__LINE__, "clGetDeviceInfo", _ert);

	cout << '\t' << name << endl;

	///////////init context/////////////////

	const cl_context_properties contextProperties [] = {
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties> (platforms[i]),
		0, 0
	};

	_ctx = clCreateContext (contextProperties, cDevice,
							devices.data (), nullptr, nullptr, &_ert);
	checkErr(__LINE__, "clCreateContext", _ert);

	_queue = clCreateCommandQueue(_ctx, devices[0], 0, &_ert);
	checkErr(__LINE__, "clCreateCommandQueue", _ert);

	// 	_cDevice = cDevice;
	_cDevice = 1;
	_devices = devices;

	cout << "Context created" << endl;
	cout << "</CreateContext>" << endl;
}

cl_kernel NN_MLP_SingleHiddenLayer::loadKernel(string fName, string kerName, string flags){

	cout << "<loadKernel>" << endl;

	ifstream in (fName);
	string code (
		(istreambuf_iterator<char> (in)),
				 istreambuf_iterator<char> ()
	);

	size_t lengths [1] = {code.length()};
	const char* codes[1] = {code.data()};

	cl_program prg = clCreateProgramWithSource(_ctx, 1,
											   codes, lengths , &_ert);
	checkErr(__LINE__, "clCreateProgramWithSource", _ert);


	if (_flags & NN_FLAG_MOMENTUM) flags += " -DUSE_MR";
	if (_flags & NN_FLAG_THRESHOLD) flags += " -DUSE_TH";

	char path[255];
	getcwd(path, 255);

	string ker_flags(path);

	ker_flags = "-I " + ker_flags;

	ker_flags +=" -DBIAS="+ to_string(_bias);
	ker_flags +=" -DCI="+ to_string(_ci);
	ker_flags +=" -DCH="+ to_string(_ch);
	ker_flags +=" -DCO="+ to_string(_co);

	ker_flags +=" " + flags;

	_ert = clBuildProgram(prg, _cDevice, _devices.data(),
						  ker_flags.c_str(),
						  nullptr, nullptr);
	// 		checkErr(__LINE__, (_ert);

	char log[4096]="No build log.";
	_ert = clGetProgramBuildInfo(prg, _devices[0], CL_PROGRAM_BUILD_LOG,
								 sizeof(log), log, nullptr);
	cout << log << endl;

	cl_kernel ker = clCreateKernel(prg, kerName.c_str(), &_ert);
	checkErr(__LINE__, "clCreateKernel", _ert);

	cout << "Loaded kernel '" + kerName + "' from '" + fName
	+ "' _with FLAGS '" + ker_flags + "'" << endl;

	cout << "</loadKernel>" << endl;

	return ker;

}


void NN_MLP_SingleHiddenLayer::bindBuffers(){
	_ai_b = clCreateBuffer(_ctx, CL_MEM_READ_ONLY,
						   sizeof(DTYPE) * _ci, nullptr, &_ert);
	_ah_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
						   sizeof(DTYPE) * _ch , nullptr, &_ert);
	_ao_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
						   sizeof(DTYPE) * _co, nullptr, &_ert);

	_wi_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
						   sizeof(DTYPE) * _ci*_ch, nullptr, &_ert);
	_wo_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
						   sizeof(DTYPE) * _ch*_co, nullptr, &_ert);

	if (_flags & NN_FLAG_MOMENTUM) {
		_pi_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
						   sizeof(DTYPE) * _ci*_ch, nullptr, &_ert);
		_po_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
						   sizeof(DTYPE) * _ch*_co, nullptr, &_ert);
	}

	if (_flags & NN_FLAG_THRESHOLD) {
		_th_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
							sizeof(DTYPE) * _ch , nullptr, &_ert);
		_to_b = clCreateBuffer(_ctx, CL_MEM_READ_WRITE,
							sizeof(DTYPE) * _co, nullptr, &_ert);
	}

	_tars_b = clCreateBuffer(_ctx, CL_MEM_READ_ONLY,
							 sizeof(DTYPE) * _co, nullptr, &_ert);
// 	_errs_b = clCreateBuffer(_ctx, CL_MEM_WRITE_ONLY,
// 							 sizeof(DTYPE) * _co, nullptr, &_ert);

	checkErr(__LINE__, "clCreateBuffer", _ert);

	_arg_idx = 0;
	_ert =  clSetKernelArg(_ker_activate, _arg_idx, sizeof(cl_mem), &_ai_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_activate, _arg_idx, sizeof(cl_mem), &_ah_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_activate, _arg_idx, sizeof(cl_mem), &_ao_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_activate, _arg_idx, sizeof(cl_mem), &_wi_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_activate, _arg_idx, sizeof(cl_mem), &_wo_b);_arg_idx++;

	if (_flags & NN_FLAG_THRESHOLD) {
		_ert |= clSetKernelArg(_ker_activate, _arg_idx, sizeof(cl_mem), &_th_b);_arg_idx++;
		_ert |= clSetKernelArg(_ker_activate, _arg_idx, sizeof(cl_mem), &_to_b);_arg_idx++;
	}

	checkErr(__LINE__, "clSetKernelArg", _ert);

	_arg_idx = 0;
	_ert =  clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_ai_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_ah_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_ao_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_wi_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_wo_b);_arg_idx++;


	if (_flags & NN_FLAG_MOMENTUM) {
		_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_pi_b);_arg_idx++;
		_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_po_b);_arg_idx++;
	}

	if (_flags & NN_FLAG_THRESHOLD) {
	_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_th_b);_arg_idx++;
	_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_to_b);_arg_idx++;
	}

	_ert |= clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(cl_mem), &_tars_b);_arg_idx++;
// 	_ert |= clSetKernelArg(_ker_backPropagation, 12, sizeof(cl_mem), &_errs_b);

	checkErr(__LINE__, "clSetKernelArg", _ert);
}


void NN_MLP_SingleHiddenLayer::setBuffers(){
	_ert = clEnqueueWriteBuffer(_queue, _ah_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ch, _ah, 0, nullptr, nullptr);
	_ert |= clEnqueueWriteBuffer(_queue, _wi_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ci*_ch, _wi, 0, nullptr, nullptr);
	_ert |= clEnqueueWriteBuffer(_queue, _wo_b, CL_TRUE, 0,
								 sizeof(DTYPE) * _ch*_co, _wo, 0, nullptr, nullptr);

	if (_flags & NN_FLAG_MOMENTUM) {
		_ert |= clEnqueueWriteBuffer(_queue, _pi_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ci*_ch, _pi, 0, nullptr, nullptr);
		_ert |= clEnqueueWriteBuffer(_queue, _po_b, CL_TRUE, 0,
								 sizeof(DTYPE) * _ch*_co, _po, 0, nullptr, nullptr);
	}

	if (_flags & NN_FLAG_THRESHOLD) {
		_ert = clEnqueueWriteBuffer(_queue, _th_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ch, _th, 0, nullptr, nullptr);
		_ert |= clEnqueueWriteBuffer(_queue, _to_b, CL_TRUE, 0,
								 sizeof(DTYPE) * _co, _to, 0, nullptr, nullptr);
	}

	_ert |= clFinish(_queue);
	checkErr(__LINE__, "setBuffers", _ert);
}

DTYPE * NN_MLP_SingleHiddenLayer::getBuffers(bool print){
	_ert = clEnqueueReadBuffer(_queue, _ao_b, CL_TRUE, 0,
							   sizeof(DTYPE) * _co, _ao, 0, nullptr, nullptr);
	_ert |= clFinish(_queue);
	checkErr(__LINE__, "getBuffers", _ert);

	if (print) {
		for (uint j=0;j<_co;j++)
			cout << _ao[j] << '\t';
		cout << endl;
	}
	return _ao;
}


void NN_MLP_SingleHiddenLayer::setupCL(
	string fName, string flags,
	cl_device_type tarDevType, int tarDevId){

	creatContext(tarDevType, tarDevId);

	_ker_activate = loadKernel(fName, "activate", flags);
	_ker_backPropagation = loadKernel(fName, "backPropagation", flags);

	static size_t defaultGSize[1] = {_ch};
	static size_t defaultLSize[1] = {_ch};

	if (gSize == nullptr) gSize = defaultGSize;
	if (lSize == nullptr) lSize = defaultLSize;

	bindBuffers();
	setBuffers();

}



NN_MLP_SingleHiddenLayer::~NN_MLP_SingleHiddenLayer(){

	cout << "<Release>" << endl;
	// 		clReleaseProgram()
	clReleaseMemObject(_ai_b);
	clReleaseMemObject(_ah_b);
	clReleaseMemObject(_ao_b);
	clReleaseMemObject(_wi_b);
	clReleaseMemObject(_wo_b);

	if (_flags & NN_FLAG_MOMENTUM) {
		clReleaseMemObject(_pi_b);
		clReleaseMemObject(_po_b);
		free(_pi);free(_po);
	}


	if (_flags & NN_FLAG_THRESHOLD) {
		clReleaseMemObject(_th_b);
		clReleaseMemObject(_to_b);
		free(_th);free(_to);
	}

	clReleaseMemObject(_tars_b);
// 	clReleaseMemObject(_errs_b);
	clReleaseKernel(_ker_activate);
	clReleaseKernel(_ker_backPropagation);
	clReleaseCommandQueue(_queue);
	clReleaseContext(_ctx);
	clReleaseDevice(_devices[0]);

	free(_ai);	free(_ah);	free(_ao);
	free(_wi);	free(_wo);
// 	free(_errs);

	cout << "Resource released!" <<endl;

	cout << "</Release>" << endl;
}

uint NN_MLP_SingleHiddenLayer::getEpochs()
{
	return _epochs;
}


void NN_MLP_SingleHiddenLayer::debugW(){
	//fetch W
	_ert = clEnqueueReadBuffer(_queue, _wi_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ci * _ch, _wi, 0, nullptr, nullptr);
	_ert |= clEnqueueReadBuffer(_queue, _wo_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ch * _co, _wo, 0, nullptr, nullptr);
	_ert |= clFinish(_queue);
	checkErr(__LINE__, "debugW", _ert);

	for (uint i=0;i<20;i++) cout<<'=';
	cout << "wi";
	for (uint i=0;i<20;i++) cout<<'=';
	cout << endl;

	for (uint i = 0;i<_ci;i++) {
		for (uint j=0;j<_ch;j++)
			cout << _wi[i*_ch + j] << '\t';
		cout << endl;
	}

	for (uint i=0;i<20;i++) cout<<'=';
	cout << "wo";
	for (uint i=0;i<20;i++) cout<<'=';
	cout << endl;

	for (uint i = 0;i<_ch;i++){
		for (uint j=0;j<_co;j++)
			cout << _wo[i*_co + j] << '\t';
		cout << endl;
	}

	for (uint i=0;i<40;i++) cout<<'=';
	cout << endl;

	if (_flags & NN_FLAG_MOMENTUM) {
		for (uint i=0;i<20;i++) cout<<'=';
		cout << "pi";
		for (uint i=0;i<20;i++) cout<<'=';
		cout << endl;

		for (uint i = 0;i<_ci;i++) {
			for (uint j=0;j<_ch;j++)
				cout << _pi[i*_ch + j] << '\t';
			cout << endl;
		}

		for (uint i=0;i<20;i++) cout<<'=';
		cout << "po";
		for (uint i=0;i<20;i++) cout<<'=';
		cout << endl;

		for (uint i = 0;i<_ch;i++){
			for (uint j=0;j<_co;j++)
				cout << _po[i*_co + j] << '\t';
			cout << endl;
		}

		for (uint i=0;i<40;i++) cout<<'=';
		cout << endl;
	}

}


void NN_MLP_SingleHiddenLayer::debugT(){
	//fetch _ah

	if (_flags & NN_FLAG_THRESHOLD) {
		_ert = clEnqueueReadBuffer(_queue, _th_b, CL_TRUE, 0,
									sizeof(DTYPE) * _ch, _th, 0, nullptr, nullptr);
		_ert |= clEnqueueReadBuffer(_queue, _to_b, CL_TRUE, 0,
									sizeof(DTYPE) * _co, _to, 0, nullptr, nullptr);
		_ert |= clFinish(_queue);
		checkErr(__LINE__, "debugT", _ert);

		for (uint i=0;i<20;i++) cout<<'=';
		cout << "th";
		for (uint i=0;i<20;i++) cout<<'=';
		cout << endl;

		for (uint j=0;j<_ch;j++)
			cout << _th[j] << '\t';
		cout << endl;

		for (uint i=0;i<20;i++) cout<<'=';
		cout << "to";
		for (uint i=0;i<20;i++) cout<<'=';
		cout << endl;

		for (uint j=0;j<_ch;j++)
			cout << _to[j] << '\t';
		cout << endl;

		for (uint i=0;i<40;i++) cout<<'=';
		cout << endl;
	}

}



void NN_MLP_SingleHiddenLayer::debugA(){
	//fetch _ah
	_ert = clEnqueueReadBuffer(_queue, _ah_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ch, _ah, 0, nullptr, nullptr);
	_ert |= clEnqueueReadBuffer(_queue, _ao_b, CL_TRUE, 0,
								sizeof(DTYPE) * _co, _ao, 0, nullptr, nullptr);
	_ert |= clFinish(_queue);
	checkErr(__LINE__, "debugA", _ert);

	for (uint i=0;i<20;i++) cout<<'=';
	cout << "ah";
	for (uint i=0;i<20;i++) cout<<'=';
	cout << endl;

	for (uint j=0;j<_ch;j++)
		cout << _ah[j] << '\t';
	cout << endl;

	for (uint i=0;i<20;i++) cout<<'=';
	cout << "ao";
	for (uint i=0;i<20;i++) cout<<'=';
	cout << endl;

	for (uint j=0;j<_co;j++)
		cout << _ao[j] << '\t';
	cout << endl;
}


void NN_MLP_SingleHiddenLayer::activate(DTYPE *inps){
	memcpy(_ai, inps, sizeof(DTYPE) * _cInp);

	_ert |= clEnqueueWriteBuffer(_queue, _ai_b, CL_TRUE, 0,
								sizeof(DTYPE) * _ci, _ai, 0, nullptr, nullptr);

	checkErr(__LINE__, "clEnqueueWriteBuffer", _ert);

	_ert = clEnqueueNDRangeKernel(_queue, _ker_activate,
									1, nullptr, gSize, lSize,
							0, nullptr, nullptr);

	checkErr(__LINE__, "clEnqueueNDRangeKernel", _ert);

	_ert = clFinish(_queue);

	checkErr(__LINE__, "activate", _ert);

}

DTYPE NN_MLP_SingleHiddenLayer::backPropagation(DTYPE *tars){

	static uint i;
	static DTYPE sum;

	getBuffers(); // CPU OP

	_ert |= clEnqueueWriteBuffer(_queue, _tars_b, CL_TRUE, 0,
								sizeof(DTYPE) * _co, tars, 0, nullptr, nullptr);

	checkErr(__LINE__, "clEnqueueWriteBuffer", _ert);

	_ert = clEnqueueNDRangeKernel(_queue, _ker_backPropagation,
									1, nullptr, gSize, lSize,
							0, nullptr, nullptr);

	sum = 0;

	// CPU OP
	static DTYPE err;

	for (i = 0 ;i<_co;i++){
		err = tars[i] - _ao[i];
		sum += err*err;
	}

	_ert |= clFinish(_queue);
	checkErr(__LINE__, "clEnqueueNDRangeKernel", _ert);


// 	// GPU Multiple
// 	_ert = clEnqueueReadBuffer(_queue, _errs_b, CL_TRUE, 0,
// 			sizeof(DTYPE) * _co , _errs, 0, nullptr, nullptr);
//
// 	_ert |= clFinish(_queue);
//
// 	checkErr(__LINE__, "backPropagation", _ert);
//
// 	sum = 0;
// 	for (i = 0 ;i<_co;i++) sum += _errs[i];

	return sum * .5;

}

DTYPE NN_MLP_SingleHiddenLayer::train(uint epochs, DTYPE lRate, DTYPE mRate){
	static float  err;
// 	_lRate = lRate;	_mRate = mRate;
	_ert = clSetKernelArg(_ker_backPropagation, _arg_idx, sizeof(DTYPE), &lRate);
	if (_flags & NN_FLAG_MOMENTUM) _ert |= clSetKernelArg(_ker_backPropagation, _arg_idx + 1, sizeof(DTYPE), &mRate);
	checkErr(__LINE__, "backPropagation", _ert);
	_epochs += epochs;
	while (epochs>0) {
		epochs -=1;
		err = 0;
		for (uint i = 0; i< cDataset; i++){
			activate(&dataset_i[i*_cInp]);
			err += backPropagation(&dataset_o[i*_co]);
		}
	}

	return err;

}
