#include <iostream>
#include <string>
// #include <iterator>
// #include <array>

#include <unistd.h>
#include <ctime>
#include <sys/time.h>
#include <math.h>

#include "nn_mlp_singlehiddenlayer.h"

using namespace std;

uint dim0 = 2, dim1 = 1;
uint step = 1000, outstep = 5000;

struct timeval t0, t1;

DTYPE fixLR = 0;
char defaultDef[] = "-DUSE_TANH";
char *extraDef = defaultDef;

void test_xor();


void test_sine(){

	NN_MLP_SingleHiddenLayer n(1, dim0, 1, 1, 0.3, 0.);
	n.setupCL("nn_mlp_singlehiddenlayer.cl", extraDef, CL_DEVICE_TYPE_CPU);
// 	n.setupCL("nn_mlp_singlehiddenlayer.cl", "-DUSE_TANH -DUSE_THRESHOLD");

	DTYPE inps[9];
	DTYPE outs[9];
	#define PI 3.14159

	for (uint i=0;i<9;i++) {
		inps[i] = 2*PI/8 * i;
		outs[i] = sin(inps[i]);
		cout << outs[i] <<'\t';
	}
	cout << endl;

	cout << "--------------------" << endl;
	n.cDataset = 9;
	n.dataset_i = inps;
	n.dataset_o = outs;

	uint countdown = 200000;
	DTYPE err = 1;
	DTYPE lr, mr, err_l0 = 4, err_l1 = 16;
	if (fixLR>0) lr = fixLR;else lr = 1./dim0;

	while ((countdown-- >0) && (err > 1e-5)) {
		err_l1 = err_l0;
		err_l0 = err;

// 		mr = lr / 4.;
		mr = 0;

		gettimeofday(&t0, nullptr);
		err = n.train(step, lr, mr);
		gettimeofday(&t1, nullptr);
		cout << ( t1.tv_sec - t0.tv_sec ) +  (t1.tv_usec - t0.tv_usec) *1e-6
			<< "\t " << n.getEpochs() << "\t"
			<< lr << "\t" << mr << "\t" << err <<"\t";

		if (n.getEpochs() % outstep == 0) {
			DTYPE xe[1];
			for (uint i=0;i<361;i++) {
				xe[0] = i*2.*PI / 360;
				n.activate(&xe[0]);
				DTYPE *res = n.getBuffers(false);
				cout << res[0] << '\t';
			}
		}
		cout <<endl;

		if (err_l0 >= err)
			lr *=1.05;
		else
			lr *=0.85;
	}

	cout << "--------------------" << endl;
	for (uint i=0;i<9;i++) {
		n.activate(&inps[i]);
		DTYPE *res = n.getBuffers(false);

		cout << res[0] << '\t';
	}
	cout <<endl;
}

void test_sin2(){


	NN_MLP_SingleHiddenLayer n(2, dim0, 1, 1, 0.3, 0.);
	n.setupCL("nn_mlp_singlehiddenlayer.cl", "-DUSE_TANH",  CL_DEVICE_TYPE_GPU);


	DTYPE inps[11*11*2];
	DTYPE outs[11*11];
	#define PI 3.14159

	for (int i=0;i<11;i++) {
		for (int j=0;j<11;j++){
			int x1 = i  * 2 - 10;
			int x2 = j  * 2 - 10;
			inps[(i*11+j)*2+0] = x1;
			inps[(i*11+j)*2+1] = x2;
			outs[i*11+j] = (x1!=0?sin(x1)/x1:1) * (x2!=0?sin(x2)/x2:1);
			cout << outs[i*11+j] <<'\t';
		}
		cout << endl;
	}
	cout << endl << endl;

	cout << "--------------------" << endl;
	n.cDataset = 121;
	n.dataset_i = inps;
	n.dataset_o = outs;

	DTYPE err = 1;
	DTYPE lr, mr, err_l0 = 2, err_l1 = 4;
	lr = .01;
	DTYPE lim, l0 = lr, l1 = lr;
	while (err > 1e-3) {
		l1 = l0;
		l0 = lr;
		err_l1 = err_l0;
		err_l0 = err;

		mr = lr / 4.;

		gettimeofday(&t0, nullptr);
		err = n.train(step, lr, mr);
		gettimeofday(&t1, nullptr);
		cout << ( t1.tv_sec - t0.tv_sec ) +  (t1.tv_usec - t0.tv_usec) *1e-6
		<< "\t " << n.getEpochs() << "\t"
		<< lr << "\t" << mr << "\t" << err <<"\t";

		if (n.getEpochs() % outstep == 0) {
			DTYPE xe[2];
			for (int i=-10;i<=10;i++) for (int j=-10;j<=10;j++){
				xe[0] = i;
				xe[1] = j;
				n.activate(&xe[0]);
				DTYPE *res = n.getBuffers(false);
				cout << res[0] << '\t';
			}
		}
		cout <<endl;

		if (err_l0 > err)
			lr *=1.05;
		else
			lr *=0.95;
	}

	cout << "--------------------" << endl;
	for (int i=0;i<11;i++) {
		for (int j=0;j<11;j++){
			n.activate(&inps[(i*11+j)*2+0]);
			DTYPE *res = n.getBuffers(false);
			cout << res[0] << '\t';
		}
		cout <<endl;
	}
	cout <<endl;
}



int main(int argc, char **argv) {
	/*
	#ifdef USE_LOGISTIC
	cout << "Using logistic" << endl;
	#else
	cout << "Using Tanh" << endl;
	#endif*/

	uint argidx = 0;

	string routine;

	argidx ++; if (argc>argidx) routine = string(argv[argidx]);
	argidx ++; if (argc>argidx) dim0 = atoi(argv[argidx]);
	argidx ++; if (argc>argidx) dim1 = atoi(argv[argidx]);
	argidx ++; if (argc>argidx) step = atoi(argv[argidx]);
	argidx ++; if (argc>argidx) outstep = atoi(argv[argidx]);
	argidx ++; if (argc>argidx) extraDef = argv[argidx];
	argidx ++; if (argc>argidx) fixLR = atof(argv[argidx]);

	cout << "DIM=" << dim0 << 'x' << dim1 <<endl;

	if (routine == "xor")
		test_xor();
	else if (routine == "sine")
		test_sine();
	else if (routine == "sin2")
		test_sin2();

}



void test_xor(){

	NN_MLP_SingleHiddenLayer n(2, dim0, 1, 0, 0.9, 0.);

	n.setupCL("nn_mlp_singlehiddenlayer.cl", extraDef, CL_DEVICE_TYPE_CPU);

	cout << "--------------------" << endl;

	DTYPE inps[] = {
		0, 0,
		0, 1,
		1, 1,
		1, 0
	};

	DTYPE outs[] = {
		0,
		1,
		1,
		0
	};

	n.cDataset = 4;
	n.dataset_i = inps;
	n.dataset_o = outs;
	// 	n.debugW();
	DTYPE err = 1;
	DTYPE lr, mr, err_l0 = 1, err_l1 = 1;
	mr = 0.;
	lr = (fixLR>0)?fixLR:0.2;
	uint cnt = 4;
	while ((cnt) &&(err > 1e-5)) {

		gettimeofday(&t0, nullptr);
		err = n.train(step, lr, mr);
		gettimeofday(&t1, nullptr);

		cout << ( t1.tv_sec - t0.tv_sec ) +  (t1.tv_usec - t0.tv_usec) *1e-6
		<< "\t " << n.getEpochs() << "\t|"
		<< lr << "\t:" << err <<endl;

// 	 		n.debugW();
// 	 		n.debugA();
		//  		n.debugT();
		// 		cnt --;
	}

	cout << "--------------------" << endl;
	n.activate(&inps[0]);n.getBuffers(true);
	n.activate(&inps[2]);n.getBuffers(true);
	n.activate(&inps[4]);n.getBuffers(true);
	n.activate(&inps[6]);n.getBuffers(true);

}
