// #include "sjgmoid.h"
#include "sigmoid.cl"


// #define USE_LOGISTIC

#ifdef USE_LOGISTIC
	#define SIGMOID logistic
	#define DSIGMOIDY dlogisticy
#endif

#ifdef USE_TANH
	#define SIGMOID tanh
	#define DSIGMOIDY dtanhy
#endif

// #define SIGMOID 1*

// global float *ai, *ah, *ao;

__kernel void activate(
	__global const float *ai, __global float *ah, __global float *ao,
	__global const float *wi, __global const float *wo

	#ifdef USE_TH
		,__global const float *th, __global const float *to
	#endif

	){
	
	uint jg = get_global_id(0); 
	
	uint i;
	
	local float sum[CH]; // CH>CO
	
	if (jg < CH - BIAS) { // 0~CH-1

		#ifdef USE_TH
			sum[jg] = th[jg];
		#else
			sum[jg] = 0;
		#endif
		
		for (i=0;i<CI;i++)
			sum[jg] += ai[i] * wi[i*CH + jg];
			
		ah[jg] = SIGMOID(sum[jg]);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (jg < CO) { // 0~CO-1

		#ifdef USE_TH
			sum[jg] = to[jg];
		#else
			sum[jg] = 0;
		#endif
		
		for (i=0;i<CH;i++)
			sum[jg] += ah[i] * wo[i*CO + jg];
		
// 		ao[jg] = SIGMOID(sum[jg]); // sigmoid for output
		ao[jg] = sum[jg];
	}
	
	
}

__kernel void backPropagation(
	__global const float *ai, __global const float *ah, __global const float *ao,
	__global float *wi, __global float *wo,

	#ifdef USE_MR
		__global float *pi, __global float *po,
	#endif

	#ifdef USE_TH
		__global float *th, __global float *to,
	#endif
	__global const float *tar,
	const float lr

	#ifdef USE_MR
		, const float mr
	#endif

	// 	,__global float *err
	){
	
	uint jg = get_global_id(0); 
	
	uint k;


	
	local float sum[CH], eh[CH], eo[CO]; // CH>CO
	
	if (jg < CO) { // 0~CO-1
		// 		eo[jg] = DSIGMOIDY(ao[jg]) * (tar[jg] - ao[jg]) * lr; // sigmoid for output
		// 		err[jg] = (tar[jg] - ao[jg]) * (tar[jg] - ao[jg]); // GPU MAC
		eo[jg] = (tar[jg] - ao[jg]) * lr;
		#ifdef USE_TH
			to[jg] += eo[jg];
		#endif
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (jg < CH) { // 0~CH-1
		sum[jg] = 0;
		for (k=0;k<CO;k++){

			#ifdef USE_MR
				wo[jg*CO + k] += eo[k] * ah[jg] + po[jg*CO + k] * mr;
				po[jg*CO + k] =  eo[k] * ah[jg];
			#else
				wo[jg*CO + k] += eo[k] * ah[jg];
			#endif
			sum[jg] += eo[k] * wo[jg*CO + k];
		}
		eh[jg] = DSIGMOIDY(ah[jg])*sum[jg] * lr;
		#ifdef USE_TH
			th[jg] += eh[jg];
		#endif
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (jg < CI) { // 0~CI-1
		for (k=0;k<CH;k++) {
			#ifdef USE_MR
				wi[jg*CH + k] += eh[k] * ai[jg] + pi[jg*CH + k] * mr;
				pi[jg*CH + k] =  eh[k] * ai[jg];
			#else
				wi[jg*CH + k] += eh[k] * ai[jg];
			#endif
		}
	}
}

