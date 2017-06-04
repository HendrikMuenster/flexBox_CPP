#ifndef flexProxDualDataHuber_H
#define flexProxDualDataHuber_H

#include "flexProx.h"

//! represents prox for a Huber data term
/*!
	\f$ \alpha\|\cdot-f\|_\epsilon \f$
*/
template<typename T>
class flexProxDualDataHuber : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	T huberEpsilon;
public:

	flexProxDualDataHuber(T aHuberEpsilon) : flexProx<T>(dualHuberProx)
	{
		huberEpsilon = aHuberEpsilon;
	}

	~flexProxDualDataHuber()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

    #ifdef __CUDACC__
	struct flexProxDualDataL1Functor
	{
		__host__ __device__
		flexProxDualDataL1Functor(T alpha) : alpha(alpha){}

		template <typename Tuple>
		__host__ __device__
        void operator()(Tuple t)
		{
			thrust::get<0>(t) = min(this->alpha, max(-this->alpha,thrust::get<1>(t) - this->alpha * thrust::get<2>(t) * thrust::get<3>(t)));
		}

		T alpha;
	};
    #endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{

	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{
		#ifdef __CUDACC__
            printf("flexProxDualDataHuber not implemented on GPU \n");
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
					T tmp = (ptrYtilde[j] - ptrSigma[j] * ptrF[j]) * alpha / (alpha + this->huberEpsilon * ptrSigma[j]);

					ptrY[j] = tmp / myMax<T>((T)1, std::abs(tmp) / alpha);
				}
			}
		#endif
	}
};

#endif
