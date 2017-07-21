#ifndef flexProxDualLInf_H
#define flexProxDualLInf_H

#include <algorithm>
#include <numeric>
#include <random>

#ifdef __CUDACC__
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/find.h>
#endif

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
private:
    std::random_device rd;
    std::mt19937 gen;

public:

    flexProxDualLInf() : flexProx<T>(dualL1IsoProx), rd(), gen(rd()) {}

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
            //project yTilde onto L1 ball with radius alpha, see: Efficient Projections onto the l1-Ball for Learning in High Dimensions, Duchi et. al.
            auto& yTilde = data->yTilde[dualNumbers[0]];
            int dim = static_cast<int>(data->yTilde[dualNumbers[0]].size());
            std::vector<int> indices(dim);
            std::iota(std::begin(indices), std::end(indices), 0); //TODO: Omp

            T s = (T)0;
            T rho = (T)0;
            while (!indices.empty())
            {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(indices.size()) - 1);
                int elem = indices[dist(gen)];
                auto partIt = std::partition(std::begin(indices), std::end(indices),
                    [&yTilde, elem](int index) {
                        return fabs(yTilde[index]) >= fabs(yTilde[elem]);
                });

                T dRho = (T)std::distance(std::begin(indices), partIt);
                T dS = std::accumulate(std::begin(indices), partIt, (T)0, [&yTilde](T lhs, int rhs) { //Omp
                    return lhs + (T)fabs(yTilde[rhs]);
                });

                if ((s + dS) - (rho + dRho) * fabs(yTilde[elem]) < alpha)
                {
                    s += dS;
                    rho += dRho;
                    indices = std::vector<int>(partIt, std::end(indices));
                }
                else
                {
                    indices = std::vector<int>(std::begin(indices), partIt);
                    auto findIt = std::find(std::begin(indices), std::end(indices), elem);
                    if(findIt != std::end(indices))
                        indices.erase(findIt);
                }
            }

            T theta = (s - alpha) / rho;
            std::transform(std::begin(yTilde), std::end(yTilde), std::begin(data->y[dualNumbers[0]]), [theta](T elem) { //omp
                return (elem > 0 ? (T)1 : (T)0) * std::max(static_cast<T>(fabs(elem)) - theta, (T)0);
            });
        }
#endif
    }

    void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
    {

    }
};

#endif
