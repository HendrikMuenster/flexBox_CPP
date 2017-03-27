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

#ifdef __CUDACC__
	struct flexProxDualDataKLFunctor
	{
		__host__ __device__
			flexProxDualDataKLFunctor(T alpha) : alpha(alpha){}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
			thrust::get<0>(t) = (T)0.5 * (this->alpha + thrust::get<1>(t) - std::sqrt(std::pow(thrust::get<1>(t) + this->alpha, (int)2) + (T)4 * (this->alpha * thrust::get<2>(t) * thrust::get<3>(t) - this->alpha * thrust::get<1>(t))));
		}

		T alpha;
};
#endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{
		#ifdef __CUDACC__
			for (int k = 0; k < dualNumbers.size(); k++)
			{
				auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[k]].begin(), data->yTilde[dualNumbers[k]].begin(), data->sigmaElt[dualNumbers[k]].begin(), fList[k].begin()));
				auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[k]].end(), data->yTilde[dualNumbers[k]].end(), data->sigmaElt[dualNumbers[k]].end(), fList[k].end()));

				thrust::for_each(startIterator, endIterator, flexProxDualDataKLFunctor(alpha));
			}
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
