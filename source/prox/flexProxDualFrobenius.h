#ifndef flexProxDualFrobenius_H
#define flexProxDualFrobenius_H

#include "flexProx.h"

//! represents prox for a Frobenius term
/*!
	\f$ \alpha\|\cdot\|_{F} \f$
*/
template<typename T>
class flexProxDualFrobenius : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:

	flexProxDualFrobenius() : flexProx<T>(dualFrobeniusProx)
	{
	}

	~flexProxDualFrobenius()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

	#ifdef __CUDACC__
		struct flexFrobeniusSquareFunctor
		{
			__host__ __device__
            flexFrobeniusSquareFunctor(){};

			__host__ __device__ T
			operator()(const T& x) const
			{
				return x * x;
			}
		};

        struct flexProxDualFrobeniusFunctor
        {
            __host__ __device__
            flexProxDualFrobeniusFunctor(T _norm) : norm(_norm){};

            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                thrust::get<0>(t) = this->norm * thrust::get<1>(t);
            }

            const T norm;
        };
    #endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
		#ifdef __CUDACC__
		    flexFrobeniusSquareFunctor unary_op;
			thrust::plus<T> binary_op;

			T norm = (T)0;
			for (int k = 0; k < dualNumbers.size(); k++)
			{
				//add sum of squared elements to norm
				norm += thrust::transform_reduce(data->yTilde[dualNumbers[k]].begin(), data->yTilde[dualNumbers[k]].end(), unary_op, (T)0, binary_op);
			}

			norm = (T)1 / std::max((T)1, std::sqrt(norm) / alpha);

			for (int k = 0; k < dualNumbers.size(); k++)
			{
                auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(data->y[dualNumbers[k]].begin(), data->yTilde[dualNumbers[k]].begin()));
                auto endIterator = thrust::make_zip_iterator(  thrust::make_tuple(data->y[dualNumbers[k]].end(),   data->yTilde[dualNumbers[k]].end()));

                thrust::for_each(startIterator,endIterator,flexProxDualFrobeniusFunctor(norm));
			}
		#else
			T norm = (T)0;
			for (int k = 0; k < dualNumbers.size(); k++)
			{
				T* ptrYTilde = data->yTilde[dualNumbers[k]].data();

				int numElements = (int)data->yTilde[dualNumbers[k]].size();

				#pragma omp parallel for reduction(+: norm)
				for (int i = 0; i < numElements; i++)
				{
					norm += ptrYTilde[i] * ptrYTilde[i];
				}
			}

			norm = (T)1 / std::max((T)1, std::sqrt(norm) / alpha);

			for (int k = 0; k < dualNumbers.size(); k++)
			{
				T* ptrY = data->y[dualNumbers[k]].data();
				T* ptrYTilde = data->yTilde[dualNumbers[k]].data();

				int numElements = (int)data->yTilde[dualNumbers[k]].size();

				#pragma omp parallel for
				for (int i = 0; i < numElements; i++)
				{
					ptrY[i] = ptrYTilde[i] * norm;
				}
			}
		#endif
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{

	}
};

#endif
