#ifndef flexProxL2Inf_H
#define flexProxL2Inf_H

#include "flexProx.h"

//! represents prox for a L2,inf non-data term
/*!
	\f$ \frac{\alpha}{2} \|\cdot\|_{2,\inf} \f$
*/
template<typename T>
class flexProxDualL2Inf : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:

    flexProxDualL2Inf() : flexProx<T>(dualL2Prox)
	{
	}

	~flexProxDualL2Inf()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

    #ifdef __CUDACC__
        /*struct flexProxDualL2Functor
        {
            __host__ __device__
            flexProxDualL2Functor(T _alpha) : alpha(_alpha){};

            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                thrust::get<0>(t) = this->alpha / (thrust::get<2>(t) + this->alpha) * thrust::get<1>(t);
            }

            const T alpha;
        };*/
    #endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
		#ifdef __CUDACC__
            /*for (int i = 0; i < dualNumbers.size(); i++)
			{
                auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[i]].begin(), data->yTilde[dualNumbers[i]].begin(), data->sigmaElt[dualNumbers[i]].begin()));
                auto endIterator = thrust::make_zip_iterator(  thrust::make_tuple(data->y[dualNumbers[i]].end(),   data->yTilde[dualNumbers[i]].end(),   data->sigmaElt[dualNumbers[i]].end()));

                thrust::for_each(startIterator,endIterator,flexProxDualL2Functor(alpha));
            }*/
		#else
            int numElements = (int)data->yTilde[dualNumbers[0]].size();
            std::vector<T> yTildeNorm(numElements, 0);
            T* ptrYTildeNorm = yTildeNorm.data();

            if (dualNumbers.size() == 1)
            {
                T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();

                #pragma omp parallel for
                for (int i = 0; i < numElements; i++)
                {
                    ptrYTildeNorm[i] = fabs(ptrYtilde0[i]);
                }
            }
            else if (dualNumbers.size() == 2)
            {
                T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();
                T* ptrYtilde1 = data->yTilde[dualNumbers[1]].data();

                #pragma omp parallel for
                for (int i = 0; i < numElements; i++)
                {
                    ptrYTildeNorm[i] = std::sqrt(ptrYtilde0[i] * ptrYtilde0[i] + ptrYtilde1[i] * ptrYtilde1[i]);
                }
            }
            else if (dualNumbers.size() == 3)
            {

                T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();
                T* ptrYtilde1 = data->yTilde[dualNumbers[1]].data();
                T* ptrYtilde2 = data->yTilde[dualNumbers[2]].data();

                #pragma omp parallel for
                for (int i = 0; i < numElements; i++)
                {
                    ptrYTildeNorm[i] = (T)sqrtf(pow2(ptrYtilde0[i]) + pow2(ptrYtilde1[i]) + pow2(ptrYtilde2[i]));
                }
            }
            else
                printf("Alert! L2,inf prox not implemented for dim>3");

            std::vector<T> sortyTildeNorm(yTildeNorm);
            std::sort(std::begin(sortyTildeNorm), std::end(sortyTildeNorm), std::greater<T>());

            T yTildeSum = 0;
            T g = -alpha;
            T lambda = 0;
            int index;
            for (index = 1; index < numElements; index++)
            {
                T lambda = sortyTildeNorm[index];
                yTildeSum += sortyTildeNorm[index - 1];
                g = yTildeSum - index * lambda - alpha;
                if (g >= 0)
                    break;
            }

            if (g < 0)
                lambda = 0;
            else
                lambda = (yTildeSum - alpha) / index;

            if (dualNumbers.size() == 1)
            {
                T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();

                T* ptrY0 = data->y[dualNumbers[0]].data();

                #pragma omp parallel for
                for (int i = 0; i < numElements; i++)
                {
                    ptrY0[i] = (ptrYTildeNorm[i] > lambda) ? ((T)1 - (lambda / ptrYTildeNorm[i])) * ptrYtilde0[i] : 0;
                }
            }
            else if (dualNumbers.size() == 2)
            {
                T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();
                T* ptrYtilde1 = data->yTilde[dualNumbers[1]].data();

                T* ptrY0 = data->y[dualNumbers[0]].data();
                T* ptrY1 = data->y[dualNumbers[1]].data();

                #pragma omp parallel for
                for (int i = 0; i < numElements; i++)
                {
                    ptrY0[i] = (ptrYTildeNorm[i] > lambda) ? ((T)1 - (lambda / ptrYTildeNorm[i])) * ptrYtilde0[i] : 0;
                    ptrY1[i] = (ptrYTildeNorm[i] > lambda) ? ((T)1 - (lambda / ptrYTildeNorm[i])) * ptrYtilde1[i] : 0;
                }
            }
            else if (dualNumbers.size() == 3)
            {
                T* ptrYtilde0 = data->yTilde[dualNumbers[0]].data();
                T* ptrYtilde1 = data->yTilde[dualNumbers[1]].data();
                T* ptrYtilde2 = data->yTilde[dualNumbers[2]].data();

                T* ptrY0 = data->y[dualNumbers[0]].data();
                T* ptrY1 = data->y[dualNumbers[1]].data();
                T* ptrY2 = data->y[dualNumbers[2]].data();

                #pragma omp parallel for
                for (int i = 0; i < numElements; i++)
                {
                    ptrY0[i] = (ptrYTildeNorm[i] > lambda) ? ((T)1 - (lambda / ptrYTildeNorm[i])) * ptrYtilde0[i] : 0;
                    ptrY1[i] = (ptrYTildeNorm[i] > lambda) ? ((T)1 - (lambda / ptrYTildeNorm[i])) * ptrYtilde1[i] : 0;
                    ptrY2[i] = (ptrYTildeNorm[i] > lambda) ? ((T)1 - (lambda / ptrYTildeNorm[i])) * ptrYtilde2[i] : 0;
                }
            }
            else
                printf("Alert! L2,inf prox not implemented for dim>3");


            
		#endif
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{

	}
};

#endif
