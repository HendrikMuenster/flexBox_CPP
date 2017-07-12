#ifndef flexTools_H
#define flexTools_H

#include <vector>
#include <numeric>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

#define VERBOSE 0

#ifdef _OPENMP
	#include <omp.h>
#endif

#ifdef __CUDACC__
	#include <thrust/device_vector.h>
	#include <thrust/transform.h>
	#include <thrust/sequence.h>
	#include <thrust/copy.h>
	#include <thrust/fill.h>
	#include <thrust/replace.h>
	#include <thrust/functional.h>
	#include <thrust/iterator/zip_iterator.h>
	#include <thrust/tuple.h>
	#include <thrust/for_each.h>
	#include <thrust/transform_reduce.h>
	#include <thrust/extrema.h>
#endif


using std::cerr;
using std::cout;
using std::endl;
using std::string;

/*! \file tools.h
	file containing global definitions and functions
*/

static const int SIGN_PLUS = 0;
static const int SIGN_MINUS = 1;
static const int SIGN_EQUALS = 2;

// Could be any number, but the whole array should fit into shared memory
#define CONST_ARRAY_SIZE 512
#define BLOCK_SIZE (64)

//! enum representing the type of concatenation
enum mySign
{
	PLUS,
	MINUS,
	EQUALS,
  COMPOSE
};

//! enum representing the type of prox
enum prox
{
    primalEmptyProx,
    dualL1AnisoProx,
    dualL1IsoProx,
    dualL2Prox,
    dualLInfProx,
	dualFrobeniusProx,
	dualHuberProx,
	dualL2DataProx,
	dualL1DataProx,
    dualLInfDataProx,
	dualKLDataProx,
	dualBoxConstraintProx,
	dualInnerProductProx
};

//! enum representing the type of a linear operator
enum linOp
{
	linearOp,
	diagonalOp,
	gradientOp,
	identityOp,
	matrixOp,
	matrixGPUOp,
	zeroOp,
	superpixelOp,
  concatOp
};

//! enum representing the type of gradient
enum gradientType
{
	forward,
	backward,
	central
};

template < typename T >
T myAbs(T x)
{
	return x > 0 ? x : -x;
}

template < typename T >
T myMin(T a, T b)
{
	return a > b ? b : a;
}

template < typename T >
T myMax(T a, T b)
{
	return a < b ? b : a;
}

double pow2(double x)
{
	return x * x;
}
float pow2(float x)
{
	return x * x;
}

template < typename T >
T vectorProduct(const std::vector<T> &v)
{
	return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}



template < typename T >
T vectorSum(const std::vector<T> &v)
{
	return std::accumulate(v.begin(), v.end(), (T)0);
}

template < typename T >
float vectorMax(std::vector<T> &v)
{
	return *std::max_element(v.begin(), v.end());
}

template < typename T >
void vectorScalarProduct(std::vector<T> &v,T scalarValue)
{
	std::transform(v.begin(), v.end(), v.begin(), [scalarValue](T x) {return scalarValue*x;});
}

template < typename T >
void vectorScalarSet(std::vector<T> &v, const T scalarValue)
{
	std::fill(v.begin(), v.end(), scalarValue);
}

template < typename T >
void vectorPlus(std::vector<T> &v1, std::vector<T> &v2)
{
	std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::plus<T>());
}

template < typename T >
void vectorMinus(std::vector<T> &v1, std::vector<T> &v2)
{
	std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::minus<T>());
}

template < typename T >
void vectorAbs(std::vector<T> &v)
{
	std::transform(v.begin(), v.end(), v.begin(), [](T x) { return std::abs(x); });
}


template < typename T >
void doOverrelaxation(std::vector<T> &x, std::vector<T> &xOld, std::vector<T> &xBar)
{
	std::transform(x.begin(), x.end(), xOld.begin(), xBar.begin(), [](T x, T y) { return x + x - y; });
}

template < typename T >
void vectorPow2(std::vector<T> &v)
{
	std::transform(v.begin(), v.end(), v.begin(), [](T x) { return x *x ; });
}

template < typename T >
void vectorAddVectorTimesVector(std::vector<T> &result, const std::vector<T> &v1, const  std::vector<T> &v2, const int signRule)
{
	switch (signRule)
	{
		case SIGN_PLUS:
		{
			int numElements = (int)result.size();
			for (int i = 0; i < numElements; ++i)
			{
				result[i] += v1[i] * v2[i];
			}
			break;
		}
		case SIGN_MINUS:
		{
			int numElements = (int)result.size();
			for (int i = 0; i < numElements; ++i)
			{
				result[i] -= v1[i] * v2[i];
			}
			break;
		}
		case SIGN_EQUALS:
		{
			int numElements = (int)result.size();
			for (int i = 0; i < numElements; ++i)
			{
				result[i] = v1[i] * v2[i];
			}
			break;
		}
	}
}

//! class for timing execution times
/*!
	Timer is a class for measuring execution times of the primal dual algorithm.
	It uses std::chrono and additionally cudaDeviceSynchronize() if
	the CUDA version of FlexBox has been compiled.
*/
class Timer
{
public:
	Timer() : t_begin(std::chrono::system_clock::now()), isStopped(false)
	{
	};

	//! resets or starts the timer
	void reset()
	{
#ifdef __CUDACC__
			cudaDeviceSynchronize();
		#endif
		t_begin = std::chrono::system_clock::now();
		isStopped = false;
	}

	//! ends the timer
	void end()
	{
#ifdef __CUDACC__
			cudaDeviceSynchronize();
		#endif
		t_end = std::chrono::system_clock::now();
		isStopped = true;
	}

	//! returns the duration
	/*!
		returns the duration between reset() and end() in seconds (as double) or 0.0 if timer has not been stopped
		\return duration in seconds
	*/
	double elapsed() const
	{
		if(isStopped)
		{
			std::chrono::duration<double> diff = t_end - t_begin;
			return diff.count();
		}
		return 0.0;

	}

private:
	bool isStopped;
	std::chrono::system_clock::time_point t_begin;
	std::chrono::system_clock::time_point t_end;
};





#ifdef __CUDACC__

	template < typename T >
	void calculateXYError(thrust::device_vector<T> &x, thrust::device_vector<T> &xOld, thrust::device_vector<T> &xError, T tau)
	{
		thrust::transform(x.begin(), x.end(), xOld.begin(), xError.begin(), (thrust::placeholders::_1 - thrust::placeholders::_2) / tau);
	}

	template < typename T >
	__host__ __device__
	T myPow2GPU(T x)
	{
		return x * x;
	}

	//! thrust functor for calculating the absolute value of vector
	template < typename T >
	struct myAbsGPU
	{
		__host__ __device__
		myAbsGPU() {}

		__host__ __device__
		T operator()(T x) const { return abs(x); }
	};

	template < typename T >
	void vectorAbs(thrust::device_vector<T> &v)
	{
		thrust::transform(v.begin(), v.end(), v.begin(), myAbsGPU<T>());
	}

	template < typename T >
	__host__ __device__
	T myMinGPU(T a, T b)
	{
		return a > b ? b : a;
	}

	__device__ float myMinGPUf(float a,float b)
	{
		return a > b ? b : a;
	}

	__device__ float myMaxGPUf(float a,float b)
	{
		return a > b ? a : b;
	}

	template < typename T >
	__host__ __device__
	T myMaxGPU(T a, T b)
	{
		return a < b ? b : a;
	}

	template < typename T >
	T vectorSum(thrust::device_vector<T> &v)
	{
		return thrust::reduce(v.begin(), v.end(), (T)0, thrust::plus<T>());
	}

	/*sets all elements in a vector to scalarValue*/
	template < typename T >
	void vectorScalarSet(thrust::device_vector<T> &v, T scalarValue)
	{
		thrust::fill(v.begin(), v.end(), scalarValue);
	}

	template < typename T >
	void vectorScalarProduct(thrust::device_vector<T> &v, const T scalarValue)
	{
		thrust::transform(v.begin(), v.end(), v.begin(), scalarValue * thrust::placeholders::_1);
	}

	template < typename T >
	void vectorMinus(thrust::device_vector<T> &v1, thrust::device_vector<T> &v2)
	{
		thrust::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), thrust::minus<T>());
	}

	template < typename T >
	void vectorAddSquared(thrust::device_vector<T> &v1, thrust::device_vector<T> &v2)
	{
		thrust::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), thrust::placeholders::_1 + thrust::placeholders::_2*thrust::placeholders::_2);
	}

	//! thrust functor for elemntwise multiplication of two vectors following a summation of the result on a third vector
	struct vectorAddVectorTimesVectorGPU
	{
		__host__ __device__
		vectorAddVectorTimesVectorGPU(const int signRule) : signRule(signRule){}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
			switch (signRule)
			{
				case SIGN_PLUS:
				{
					thrust::get<0>(t) += thrust::get<1>(t) * thrust::get<2>(t);
					break;
				}
				case SIGN_MINUS:
				{
					thrust::get<0>(t) -= thrust::get<1>(t) * thrust::get<2>(t);
					break;
				}
				case SIGN_EQUALS:
				{
					thrust::get<0>(t) = thrust::get<1>(t) * thrust::get<2>(t);
					break;
				}
			}
		}

		const int signRule;
	};

	template < typename T >
	void vectorAddVectorTimesVector(thrust::device_vector<T> &result, const thrust::device_vector<T> &v1, const  thrust::device_vector<T> &v2, const int signRule)
	{
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(result.begin(), v1.begin(), v2.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(result.end(), v1.end(), v2.end())),
			vectorAddVectorTimesVectorGPU(signRule));
	}


	template < typename T >
	__host__ __device__
	T sqrtGPU(T x)
	{
		return sqrt(x);
	}

	template < typename T >
	void vectorSqrt(thrust::device_vector<T> &v1)
	{
		thrust::transform(v1.begin(), v1.end(), v1.begin(), sqrtGPU<T>());
	}

	template < typename T >
	float vectorMax(thrust::device_vector<T> &v)
	{
		return *thrust::max_element(v.begin(), v.end());
	}




#endif


	//kernels for CUDA operators

#ifdef __CUDACC__
/*
// cuda error checking
std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		std::ofstream out("output.txt");
		out << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
		if (prev_line>0) out << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
		out.close();

		cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
		if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
		system("pause");
		exit(1);
	}
	prev_file = file;
	prev_line = line;
}

void writeOutput(char* writeString)
{

	std::ofstream out("log.txt");
	out << endl << writeString << endl;
	out.close();
}

#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
//#if DO_CUDA_CHECK
//	#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
//#else
//	#define CUDA_CHECK 0
//#endif
*/

#endif



#endif
