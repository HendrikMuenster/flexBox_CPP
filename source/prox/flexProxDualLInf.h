#ifndef flexProxDualLInf_H
#define flexProxDualLInf_H

#include <algorithm>
#include <numeric>

#include "flexProx.h"


//! represents prox for a LInf non-data term
/*!
\f$ \alpha \|\cdot\|_{\infty} \f$
*/
template<typename T>
class flexProxDualLInf : public flexProx<T>
{

#ifdef __CUDACC__
    typedef thrust::device_vector<T> Tdata;
#else
    typedef std::vector<T> Tdata;
#endif

public:

    flexProxDualLInf() : flexProx<T>(dualL1IsoProx) {}

    ~flexProxDualLInf()
    {
        if (VERBOSE > 0) printf("Destructor prox\n!");
    }

#ifdef __CUDACC__
    /*struct flexProxDualL1IsoDim1Functor
    {
        __host__ __device__
            flexProxDualL1IsoDim1Functor(T alpha) : alpha(alpha) {}

        template <typename Tuple>
        __host__ __device__
            void operator()(Tuple t)
        {
            thrust::get<0>(t) = thrust::get<1>(t) / max((T)1, fabs(thrust::get<1>(t)) / this->alpha);
        }

        T alpha;
    };

    struct flexProxDualL1IsoDim2Functor
    {
        __host__ __device__
            flexProxDualL1IsoDim2Functor(T alpha) : alpha(alpha) {}

        template <typename Tuple>
        __host__ __device__
            void operator()(Tuple t)
        {
            T norm = max((T)1, sqrt(pow(thrust::get<2>(t), (int)2) + pow(thrust::get<3>(t), (int)2)) / this->alpha);

            thrust::get<0>(t) = thrust::get<2>(t) / norm;
            thrust::get<1>(t) = thrust::get<3>(t) / norm;
        }

        T alpha;
    };

    struct flexProxDualL1IsoDim3Functor
    {
        __host__ __device__
            flexProxDualL1IsoDim3Functor(T alpha) : alpha(alpha) {}

        template <typename Tuple>
        __host__ __device__
            void operator()(Tuple t)
        {
            T norm = max((T)1, sqrt(pow(thrust::get<3>(t), (int)2) + pow(thrust::get<4>(t), (int)2) + pow(thrust::get<5>(t), (int)2)) / this->alpha);

            thrust::get<0>(t) = thrust::get<3>(t) / norm;
            thrust::get<1>(t) = thrust::get<4>(t) / norm;
            thrust::get<2>(t) = thrust::get<5>(t) / norm;
        }

        T alpha;
    };*/
#endif

    void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
    {
#ifdef __CUDACC__
        /*if (dualNumbers.size() == 1)
        {
            auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].begin(), data->yTilde[dualNumbers[0]].begin()));
            auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].end(), data->yTilde[dualNumbers[0]].end()));

            thrust::for_each(startIterator, endIterator, flexProxDualL1IsoDim1Functor(alpha));
        }
        else if (dualNumbers.size() == 2)
        {
            auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].begin(), data->y[dualNumbers[1]].begin(), data->yTilde[dualNumbers[0]].begin(), data->yTilde[dualNumbers[1]].begin()));
            auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].end(), data->y[dualNumbers[1]].end(), data->yTilde[dualNumbers[0]].end(), data->yTilde[dualNumbers[1]].end()));

            thrust::for_each(startIterator, endIterator, flexProxDualL1IsoDim2Functor(alpha));
        }
        else if (dualNumbers.size() == 3)
        {
            auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].begin(), data->y[dualNumbers[1]].begin(), data->y[dualNumbers[2]].begin(), data->yTilde[dualNumbers[0]].begin(), data->yTilde[dualNumbers[1]].begin(), data->yTilde[dualNumbers[2]].begin()));
            auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].end(), data->y[dualNumbers[1]].end(), data->y[dualNumbers[2]].end(), data->yTilde[dualNumbers[0]].end(), data->yTilde[dualNumbers[1]].end(), data->yTilde[dualNumbers[2]].end()));

            thrust::for_each(startIterator, endIterator, flexProxDualL1IsoDim3Functor(alpha));
        }
        else
        {
            printf("Alert! Iso prox not implemented in CUDA for dim>3");
        }*/
#else
        if(dualNumbers.size() != 1)
            printf("Alert! LInf prox only defined for dim = 1");
        else
        {
            T* ptrYtilde = data->yTilde[dualNumbers[0]].data();
            T* ptrY = data->y[dualNumbers[0]].data();
            size_t ySize = data->yTilde[dualNumbers[0]].size();

            
            T norm = 0;
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(ySize); i++)
            {
                norm += fabs(ptrYtilde[i]);
            }

            if (norm < alpha)
            {
                data->y[dualNumbers[0]] = data->yTilde[dualNumbers[0]];
                return;
            }

            
            auto yTildeSort = data->yTilde[dualNumbers[0]];
            std::sort(std::begin(yTildeSort), std::end(yTildeSort), [](T lhs, T rhs) { return fabs(lhs) > fabs(rhs); });
            Tdata cumSum = Tdata(yTildeSort.size());

            T sum = 0;
            auto yTildeSortPtr = yTildeSort.data();

            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(ySize); i++)
            {
                sum += yTildeSortPtr[i];
                cumSum[i] = sum - alpha;
            }

            //find maximal rho s.t. yTildeSort(rho) > cumSum(rho)/rho
            //size_t rho = 0;
            T theta = 0;
            auto cumSumPtr = cumSum.data();
            #pragma omp parallel for
            for (int i = static_cast<int>(ySize) - 1; i >= 0; i--)
            {
                if (yTildeSortPtr[i] > cumSumPtr[i] / (i + 1))
                {
                    //rho = i;
                    theta = cumSumPtr[i] / (i + 1);
                    break;
                }
            }

            theta = std::max(static_cast<T>(0), theta);

            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(ySize); i++)
            {
                ptrY[i] = (ptrYtilde[i] > 0 ? 1 : -1) * std::max(static_cast<T>(0), static_cast<T>(fabs(ptrYtilde[i])) - theta);
                
            }


        }
#endif
    }

    void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
    {

    }
};

#endif
