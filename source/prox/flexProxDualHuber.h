#ifndef flexProxDualHuber_H
#define flexProxDualHuber_H

#include "flexProx.h"

template<typename T>
class flexProxDualHuber : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	T huberEpsilon;
public:

	flexProxDualHuber(T _huberEpsilon) : flexProx<T>(dualHuberProx)
	{
		huberEpsilon = _huberEpsilon;
	}

	~flexProxDualHuber()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

    #ifdef __CUDACC__
	struct flexProxDualHuberDim2Functor
	{
		__host__ __device__
		flexProxDualHuberDim2Functor(T _epsi, T _alpha) : epsi(_epsi), alpha(_alpha), epsiAlpha(epsi / alpha){}

		template <typename Tuple>
		__host__ __device__
        void operator()(Tuple t)
		{
            T huberFactor1 = (T)1 / ((T)1 + thrust::get<4>(t) * epsiAlpha);
            T huberFactor2 = (T)1 / ((T)1 + thrust::get<5>(t) * epsiAlpha);

			T norm = max((T)1, sqrt( pow(thrust::get<2>(t)*huberFactor1,(int)2) + pow(thrust::get<3>(t)*huberFactor2,(int)2)) / alpha);

			thrust::get<0>(t) = thrust::get<2>(t) * huberFactor1 / norm;
			thrust::get<1>(t) = thrust::get<3>(t) * huberFactor2 / norm;
		}

        const T epsi;
		const T alpha;
        const T epsiAlpha;
	};

	struct flexProxDualHuberDim3Functor
	{
		__host__ __device__
		flexProxDualHuberDim3Functor(T _epsi, T _alpha) : epsi(_epsi), alpha(_alpha), epsiAlpha(epsi / alpha){}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
			T huberFactor1 = (T)1 / ((T)1 + thrust::get<6>(t) * epsiAlpha);
			T huberFactor2 = (T)1 / ((T)1 + thrust::get<7>(t) * epsiAlpha);
			T huberFactor3 = (T)1 / ((T)1 + thrust::get<8>(t) * epsiAlpha);

			T norm = max((T)1, sqrt( pow(thrust::get<3>(t)*huberFactor1,(int)2) + pow(thrust::get<4>(t)*huberFactor2,(int)2) + pow(thrust::get<5>(t)*huberFactor3,(int)2)) / alpha);

			thrust::get<0>(t) = thrust::get<3>(t) * huberFactor1 / norm;
			thrust::get<1>(t) = thrust::get<4>(t) * huberFactor2 / norm;
			thrust::get<2>(t) = thrust::get<5>(t) * huberFactor3 / norm;
		}

		const T epsi;
		const T alpha;
		const T epsiAlpha;
	};
    #endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
		#ifdef __CUDACC__
            if (dualNumbers.size() == 2)
			{
                auto startIterator = thrust::make_zip_iterator( thrust::make_tuple(data->y[dualNumbers[0]].begin(), data->y[dualNumbers[1]].begin(), data->yTilde[dualNumbers[0]].begin(), data->yTilde[dualNumbers[1]].begin(), data->sigmaElt[dualNumbers[0]].begin(), data->sigmaElt[dualNumbers[1]].begin()));
                auto endIterator =   thrust::make_zip_iterator( thrust::make_tuple(data->y[dualNumbers[0]].end(),   data->y[dualNumbers[1]].end(),   data->yTilde[dualNumbers[0]].end(),   data->yTilde[dualNumbers[1]].end(),   data->sigmaElt[dualNumbers[0]].end(),   data->sigmaElt[dualNumbers[1]].end()));

                thrust::for_each(startIterator,endIterator,flexProxDualHuberDim2Functor(this->huberEpsilon,alpha));
			}
			else if (dualNumbers.size() == 3) 
            {
				auto startIterator = thrust::make_zip_iterator( thrust::make_tuple(data->y[dualNumbers[0]].begin(), data->y[dualNumbers[1]].begin(), data->y[dualNumbers[2]].begin(), data->yTilde[dualNumbers[0]].begin(), data->yTilde[dualNumbers[1]].begin(), data->yTilde[dualNumbers[2]].begin(), data->sigmaElt[dualNumbers[0]].begin(), data->sigmaElt[dualNumbers[1]].begin(), data->sigmaElt[dualNumbers[2]].begin()));
				auto endIterator =   thrust::make_zip_iterator( thrust::make_tuple(data->y[dualNumbers[0]].end(),   data->y[dualNumbers[1]].end(),	 data->y[dualNumbers[2]].end(),	  data->yTilde[dualNumbers[0]].end(),   data->yTilde[dualNumbers[1]].end(),   data->yTilde[dualNumbers[2]].end(),   data->sigmaElt[dualNumbers[0]].end(),   data->sigmaElt[dualNumbers[1]].end(),   data->sigmaElt[dualNumbers[2]].end()));

				thrust::for_each(startIterator, endIterator, flexProxDualHuberDim3Functor(this->huberEpsilon, alpha));
            }
			else
			{
				printf("Alert! Huber prox not implemented in CUDA for dim!={2,3}\n");
			}
		#else
			if (dualNumbers.size() == 1)
			{
				T* ptrY0 = data->y[dualNumbers[0]].data();

				T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();

				T* ptrSigma = data->sigmaElt[dualNumbers[0]].data();

				int numElements = (int)data->yTilde[dualNumbers[0]].size();

				#pragma omp parallel for
				for (int i = 0; i < numElements; i++)
				{
					T huberFactor = alpha / (alpha + ptrSigma[i] * this->huberEpsilon);

					T yTmp = huberFactor / myMax<T>((T)1, huberFactor*std::abs(ptrYtilde0[i]) / alpha);

					ptrY0[i] = ptrYtilde0[i] * yTmp;
				}
			}
			else if (dualNumbers.size() == 2)
			{
				T* ptrY0 = data->y[dualNumbers[0]].data();
				T* ptrY1 = data->y[dualNumbers[1]].data();

				T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();
				T* ptrYtilde1 = data->yTilde[dualNumbers[1]].data();

				T* ptrSigma = data->sigmaElt[dualNumbers[0]].data();

				int numElements = (int)data->yTilde[dualNumbers[0]].size();

				#pragma omp parallel for
				for (int i = 0; i < numElements; i++)
				{
					T huberFactor = alpha / (alpha + ptrSigma[i] * this->huberEpsilon);

					T yTmp = huberFactor / myMax<T>((T)1, huberFactor*std::sqrt(pow2(ptrYtilde0[i]) + pow2(ptrYtilde1[i])) / alpha);

					ptrY0[i] = ptrYtilde0[i] * yTmp;
					ptrY1[i] = ptrYtilde1[i] * yTmp;
				}
			}
			else if (dualNumbers.size() == 3)
			{
				T* ptrY0 = data->y[dualNumbers[0]].data();
				T* ptrY1 = data->y[dualNumbers[1]].data();
				T* ptrY2 = data->y[dualNumbers[2]].data();

				T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();
				T* ptrYtilde1 = data->yTilde[dualNumbers[1]].data();
				T* ptrYtilde2 = data->yTilde[dualNumbers[2]].data();

				T* ptrSigma = data->sigmaElt[dualNumbers[0]].data();

				int numElements = (int)data->yTilde[dualNumbers[0]].size();

				#pragma omp parallel for
				for (int i = 0; i < numElements; i++)
				{
					T huberFactor = alpha / (alpha + ptrSigma[i] * this->huberEpsilon);

					T yTmp = huberFactor / std::max((T)1, huberFactor * std::sqrt(pow2(ptrYtilde0[i]) + pow2(ptrYtilde1[i]) + pow2(ptrYtilde2[i])) / alpha);

					ptrY0[i] = ptrYtilde0[i] * yTmp;
					ptrY1[i] = ptrYtilde1[i] * yTmp;
					ptrY2[i] = ptrYtilde2[i] * yTmp;
				}
			}
			else
			{
				printf("Alert! Huber prox not implemented for dim>3");
			}
		#endif
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{

	}
};

#endif
