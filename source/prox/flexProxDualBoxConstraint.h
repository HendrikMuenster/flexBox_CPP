#ifndef flexProxDualBoxConstraint_H
#define flexProxDualBoxConstraint_H

#include "flexProx.h"

//! represents prox for a box constraint
/*!
	\f$ \delta_{\{\bar{u} : u_{1}\leq \bar{u}\leq u_{2} \}}(\cdot) \f$
*/
template <typename T>
class flexProxDualBoxConstraint : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	T minVal;
	T maxVal;
public:

	//! initializes the box constraint prox
	/*!
		\param aMinval lower bound (equals \f$u_1\f$)
		\param aMaxVal upper bound (equals \f$u_2\f$)
	*/
	flexProxDualBoxConstraint(T aMinVal, T aMaxVal) : flexProx<T>(dualBoxConstraintProx)
	{
		minVal = aMinVal;
		maxVal = aMaxVal;
	}

	~flexProxDualBoxConstraint()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

    #ifdef __CUDACC__
	struct flexProxDualBoxConstraintFunctor
	{
		__host__ __device__
		flexProxDualBoxConstraintFunctor(T aMinVal, T maxVal) : minVal(aMinVal), maxVal(maxVal){}

		template <typename Tuple>
		__host__ __device__
        void operator()(Tuple t)
		{
			thrust::get<0>(t) = min((T)0, thrust::get<1>(t) - thrust::get<2>(t) * this->minVal) + max((T)0, thrust::get<1>(t) - thrust::get<2>(t) * this->maxVal);
		}

		const T minVal;
        const T maxVal;
	};
    #endif

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
		#ifdef __CUDACC__
            for (int k = 0; k < dualNumbers.size(); k++)
            {
                auto startIterator = thrust::make_zip_iterator( thrust::make_tuple(data->y[dualNumbers[k]].begin(), data->yTilde[dualNumbers[k]].begin(), data->sigmaElt[dualNumbers[k]].begin()));
                auto endIterator =   thrust::make_zip_iterator( thrust::make_tuple(data->y[dualNumbers[k]].end(),   data->yTilde[dualNumbers[k]].end(),   data->sigmaElt[dualNumbers[k]].end()));

                thrust::for_each(startIterator,endIterator,flexProxDualBoxConstraintFunctor(this->minVal,this->maxVal));
            }
		#else
			for (int k = 0; k < dualNumbers.size(); k++)
			{
				T* ptrY = data->y[dualNumbers[k]].data();
				T* ptrYtilde = data->yTilde[dualNumbers[k]].data();
				T* ptrSigma = data->sigmaElt[dualNumbers[k]].data();

				int numElements = (int)data->yTilde[dualNumbers[k]].size();

				#pragma omp parallel for
				for (int i = 0; i < numElements; i++)
				{
					ptrY[i] = myMax<T>((T)0, ptrYtilde[i] - ptrSigma[i] * this->maxVal) + myMin<T>((T)0, ptrYtilde[i] - ptrSigma[i] * this->minVal);
					//if (i == 54)printf("%f\n", ptrY[i]);
				}
			}
		#endif
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{

	}
};

#endif
