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
        struct AbsFunctor
        {
            __host__ __device__
                T operator()(const T& x) const
            {
                return (T)abs(x);
            }
        };
        struct L21NormDim2
        {
            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                thrust::get<0>(t) = sqrt(pow(thrust::get<1>(t),2) + pow(thrust::get<2>(t),2));
            }
        };

        struct L21NormDim3
        {
            template <typename Tuple>
            __host__ __device__
                void operator()(Tuple t)
            {
                thrust::get<0>(t) = sqrt(pow(thrust::get<1>(t), 2) + pow(thrust::get<2>(t), 2) + pow(thrust::get<3>(t), 2));
            }
        };

        struct FindTransform
        {
            FindTransform(T aAlpha) : alpha(aAlpha) { }
            template <typename Tuple>
            __host__ __device__
                void operator()(Tuple t)
            {
                thrust::get<0>(t) = thrust::get<1>(t) - thrust::get<2>(t) * thrust::get<3>(t) - alpha;
            }
            T alpha;
        };

        struct GreaterEqualZero
        {
            __host__ __device__
                bool operator()(T val)
            {
                return (val >= (T)0);
            }
        };


        struct ResultFunctor
        {
            ResultFunctor(T aLambda) : lambda(aLambda) { }
            template <typename Tuple>
            __host__ __device__
                void operator()(Tuple t)
            {
                thrust::get<0>(t) = (thrust::get<1>(t) > lambda) ? ((T)1 - (lambda / thrust::get<1>(t))) * thrust::get<2>(t) : 0;
            }
            T lambda;
        };

    #endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
        int numElements = (int)data->yTilde[dualNumbers[0]].size();

		#ifdef __CUDACC__
            Tdata yTildeNorm(numElements);
            if (dualNumbers.size() == 1)
            {
                auto startIterator = data->yTilde[dualNumbers[0]].begin();
                auto endIterator = data->yTilde[dualNumbers[0]].end();

                thrust::transform(startIterator, endIterator, yTildeNorm.begin(), AbsFunctor());
            }
            else if (dualNumbers.size() == 2)
            {
                auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(yTildeNorm.begin(), data->yTilde[dualNumbers[0]].begin(), data->yTilde[dualNumbers[1]].begin()));
                auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(yTildeNorm.end(), data->yTilde[dualNumbers[0]].end(), data->yTilde[dualNumbers[1]].end()));

                thrust::for_each(startIterator, endIterator, L21NormDim2());
            }
            else if (dualNumbers.size() == 3)
            {
                auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(yTildeNorm.begin(), data->yTilde[dualNumbers[0]].begin(), data->yTilde[dualNumbers[1]].begin(), data->yTilde[dualNumbers[2]].begin()));
                auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(yTildeNorm.end(), data->yTilde[dualNumbers[0]].end(), data->yTilde[dualNumbers[1]].end(), data->yTilde[dualNumbers[2]].end()));

                thrust::for_each(startIterator, endIterator, L21NormDim3());
            }
            else
            {
                printf("Alert! Iso prox not implemented in CUDA for dim>3");
            }

            Tdata sortyTildeNorm(yTildeNorm);
            thrust::sort(sortyTildeNorm.begin(), sortyTildeNorm.end(), thrust::greater<T>());

            Tdata yTildeSum(yTildeNorm.size());
            Tdata g(yTildeNorm.size());
            thrust::exclusive_scan(sortyTildeNorm.begin(), sortyTildeNorm.end(), yTildeSum.begin());
            thrust::counting_iterator<int> first(0);
            thrust::counting_iterator<int> last(yTildeNorm.size());

            auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(g.begin(), yTildeSum.begin(), first, sortyTildeNorm.begin()));
            auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(g.end(), yTildeSum.end(), last, sortyTildeNorm.end()));

            thrust::for_each(startIterator, endIterator, FindTransform(alpha));

            auto findIter = g.begin();
            findIter = thrust::find_if(g.begin(), g.end(), GreaterEqualZero());
            T lambda = 0;
            int index = -1;
            if (findIter != g.end() && findIter != g.begin())
            {
                index = thrust::distance(g.begin(), findIter);
                lambda = (yTildeSum[index] - alpha) / index;
            }

            for (int k = 0; k < dualNumbers.size(); k++)
            {
                auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[k]].begin(), yTildeNorm.begin(), data->yTilde[dualNumbers[k]].begin()));
                auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[k]].end(), yTildeNorm.end(), data->yTilde[dualNumbers[k]].end()));

                thrust::for_each(startIterator, endIterator, ResultFunctor(lambda));
            }

		#else

            std::vector<T> yTildeNorm(numElements);
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

			#pragma omp parallel for
			for (int k = 0; k < dualNumbers.size(); k++)
            {
				T* ptrY = data->y[dualNumbers[k]].data();
				T* ptrYtilde = data->yTilde[dualNumbers[k]].data();

				#pragma omp parallel for
                for (int i = 0; i < numElements; i++)
                {
                    ptrY[i] = (ptrYTildeNorm[i] > lambda) ? ((T)1 - (lambda / ptrYTildeNorm[i])) * ptrYtilde[i] : 0;
                }
            }
		#endif
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{

	}
};

#endif
