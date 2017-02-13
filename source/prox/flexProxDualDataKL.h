#ifndef flexProxDualKL_H
#define flexProxDualKL_H

#include "flexProx.h"

template<typename T>
class flexProxDualDataKL : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:

	flexProxDualDataKL() : flexProx<T>(dualKLDataProx)
	{
	}

	~flexProxDualDataKL()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{

	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{
		#ifdef __CUDACC__
            printf("flexProxDualDataKL Prox not implemented for CUDA\n");
		#else
			for (int i = 0; i < dualNumbers.size(); i++)
			{
				T* ptrY = data->y[dualNumbers[i]].data();
				T* ptrYtilde = data->yTilde[dualNumbers[i]].data();
				T* ptrSigma = data->sigmaElt[dualNumbers[i]].data();

				T* ptrF = fList[i].data();

				int numElements = (int)data->yTilde[dualNumbers[i]].size();

				#pragma omp parallel for
				for (int j = 0; j < numElements; j++)
				{
					ptrY[j] = (T)0.5 * (alpha + ptrYtilde[j] - std::sqrt(std::pow(ptrYtilde[j] + alpha, (int)2) + (T)4 * (alpha * ptrSigma[j] * ptrF[j] - alpha * ptrYtilde[j])));
				}
			}
		#endif
	}
};

#endif
