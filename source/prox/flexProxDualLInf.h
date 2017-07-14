#ifndef flexProxDualLInf_H
#define flexProxDualLInf_H

#include <algorithm>
#include <numeric>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/find.h>

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

    struct CompareFunctor
    {
	__host__ __device__
	bool operator()(const T& lhs, const T& rhs)
	{
		return abs(lhs) > abs(rhs);
	}
    };
    struct UpdateYFunctor
    {
        __host__ __device__
            UpdateYFunctor(T aTheta) : theta(aTheta) {}

        template <typename Tuple>
        __host__ __device__
            void operator()(Tuple t)
        {
            thrust::get<0>(t) = (thrust::get<1>(t) > 0 ? 1 : -1) * max((T)0, abs(thrust::get<1>(t)) - theta);
        }

        T theta;
    };

    struct FindFunctor
    {
        __host__ __device__
            FindFunctor(T aAlpha) : alpha(aAlpha) {}

        template <typename Tuple>
        __host__ __device__
            bool operator()(Tuple t)
        {
            return thrust::get<0>(t) > (thrust::get<1>(t) - alpha) / thrust::get<2>(t);
        }

        T alpha;
    };

    struct AbsFunctor
    {
        __host__ __device__
            T operator()(const T& x) const
        {
            return (T)abs(x);
        }
    };
#endif

    void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
    {
#ifdef __CUDACC__
        if (dualNumbers.size() != 1)
        {
            printf("Alert! LInf prox only defined for dim = 1");
        }
        else
        {
           T norm = (T)thrust::transform_reduce(data->yTilde[dualNumbers[0]].begin(), data->yTilde[dualNumbers[0]].end(), AbsFunctor(), (T)0, thrust::plus<T>());
            
		if (norm < alpha)
            {
                data->y[dualNumbers[0]] = data->yTilde[dualNumbers[0]];
                return;
            }

            Tdata yTildeSort(data->yTilde[dualNumbers[0]]);
            thrust::sort(yTildeSort.begin(), yTildeSort.end(), CompareFunctor());
            Tdata cumSum(yTildeSort);
            thrust::inclusive_scan(yTildeSort.begin(), yTildeSort.end(), cumSum.begin());

            thrust::device_vector<T> seq(data->y[dualNumbers[0]].size());
            thrust::sequence(seq.begin(), seq.end(), 1);

            auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(yTildeSort.rbegin(), cumSum.rbegin(), seq.rbegin()));
            auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(yTildeSort.rend(), cumSum.rend(), seq.rend()));

            auto findIt = thrust::find_if(startIterator, endIterator, FindFunctor(alpha));
            auto derefIt = *findIt;
            T theta = (thrust::get<1>(derefIt) - alpha) / thrust::get<2>(derefIt);
            theta = std::max(static_cast<T>(0), theta);

            auto startIterator2 = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].begin(), data->yTilde[dualNumbers[0]].begin()));
            auto endIterator2 = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[0]].end(), data->yTilde[dualNumbers[0]].end()));

            thrust::for_each(startIterator2, endIterator2, UpdateYFunctor(theta));
        }
        
#else
        if(dualNumbers.size() != 1)
            printf("Alert! LInf prox only defined for dim = 1");
        else
        {
            T* ptrYtilde = data->yTilde[dualNumbers[0]].data();
            T* ptrY = data->y[dualNumbers[0]].data();
            size_t ySize = data->yTilde[dualNumbers[0]].size();

            
            T norm = 0;
            #pragma omp parallel for reduction (+:norm)
            for (int i = 0; i < static_cast<int>(ySize); i++)
                norm += fabs(ptrYtilde[i]);

            if (norm < alpha)
            {
                data->y[dualNumbers[0]] = data->yTilde[dualNumbers[0]];
                return;
            }

            
            auto yTildeSort = data->yTilde[dualNumbers[0]]; //copy because it needs to be sorted
            std::sort(std::begin(yTildeSort), std::end(yTildeSort), [](T lhs, T rhs) { return fabs(lhs) > fabs(rhs); }); //sort abs descending
            Tdata cumSum = Tdata(yTildeSort.size());
            std::partial_sum(std::begin(yTildeSort), std::end(yTildeSort), std::begin(cumSum)); //no omp because it has data dependencies

            T sum = 0;
            auto yTildeSortPtr = yTildeSort.data();

            //find maximal index s.t. yTildeSort(rho) > (cumSum(rho) - alpha)/rho
            T theta = 0;
            auto cumSumPtr = cumSum.data();
            #pragma omp parallel for
            for (int i = static_cast<int>(ySize) - 1; i >= 0; i--)
            {
                if (yTildeSortPtr[i] > (cumSumPtr[i] - alpha)/ (i + 1))
                {
                    theta = (cumSumPtr[i] - alpha) / (i + 1);
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
