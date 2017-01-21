#ifndef flexDiagonalOperator_H
#define flexDiagonalOperator_H

#include <vector>
#include "flexLinearOperator.h"

template <typename T>
class flexDiagonalOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	Tdata diagonalElements;
public:

	flexDiagonalOperator(std::vector<T> aDiagonalElements)
	: flexLinearOperator<T>(static_cast<int>(aDiagonalElements.size()), static_cast<int>(aDiagonalElements.size()), diagonalOp)
	{
		this->diagonalElements.resize(aDiagonalElements.size());

		#ifdef __CUDACC__
			thrust::copy(aDiagonalElements.begin(), aDiagonalElements.end(), this->diagonalElements.begin());

		#else
			this->diagonalElements = aDiagonalElements;
		#endif
	}

	#ifdef __CUDACC__
		flexDiagonalOperator(Tdata aDiagonalElements) : diagonalElements(aDiagonalElements), flexLinearOperator<T>(static_cast<int>(aDiagonalElements.size()), static_cast<int>(aDiagonalElements.size()), diagonalOp)
		{

		};
	#endif

	flexDiagonalOperator<T>* copy()
	{
		flexDiagonalOperator<T>* A = new flexDiagonalOperator<T>(this->diagonalElements);

		return A;
	}

  #ifdef __CUDACC__
	  struct flexDiagonalOperatorTimesFunctor
		{
			__host__ __device__
			flexDiagonalOperatorTimesFunctor(){}

			template <typename Tuple>
			__host__ __device__
			void operator()(Tuple t)
			{
	            thrust::get<0>(t) = thrust::get<1>(t) * thrust::get<2>(t);
			}
		};
  #endif

	//apply linear operator to vector
	void times(const Tdata &input, Tdata &output)
	{
        #ifdef __CUDACC__
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(output.begin(), input.begin(), this->diagonalElements.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(output.end(),   input.end(),   this->diagonalElements.end())),
				flexDiagonalOperatorTimesFunctor());
        #else
            int numElements = (int)output.size();

            #pragma omp parallel for
            for (int i = 0; i < numElements; ++i)
            {
                output[i] = input[i] * this->diagonalElements[i];
            }
        #endif
	}

  #ifdef __CUDACC__
  struct flexDiagonalOperatorTimesPlusFunctor
	{
		__host__ __device__
		flexDiagonalOperatorTimesPlusFunctor(){}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
            thrust::get<0>(t) += thrust::get<1>(t) * thrust::get<2>(t);
		}
	};
  #endif

	void timesPlus(const Tdata &input, Tdata &output)
	{
		#ifdef __CUDACC__
			thrust::for_each(
			    thrust::make_zip_iterator(thrust::make_tuple(output.begin(), input.begin(), this->diagonalElements.begin())),
			    thrust::make_zip_iterator(thrust::make_tuple(output.end(),   input.end(),   this->diagonalElements.end())),
				flexDiagonalOperatorTimesPlusFunctor());
    #else
	    int numElements = (int)output.size();

	    #pragma omp parallel for
	    for (int i = 0; i < numElements; ++i)
	    {
	        output[i] += input[i] * this->diagonalElements[i];
	    }
		#endif
	}

  #ifdef __CUDACC__
	  struct flexDiagonalOperatorTimesMinusFunctor
		{
			__host__ __device__
			flexDiagonalOperatorTimesMinusFunctor(){}

			template <typename Tuple>
			__host__ __device__
			void operator()(Tuple t)
			{
				thrust::get<0>(t) -= thrust::get<1>(t) * thrust::get<2>(t);
			}
		};
  #endif

	void timesMinus(const Tdata &input, Tdata &output)
	{
        #ifdef __CUDACC__
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(output.begin(), input.begin(), this->diagonalElements.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(output.end(),   input.end(),   this->diagonalElements.end())),
				flexDiagonalOperatorTimesPlusFunctor());
        #else
            int numElements = (int)output.size();

            #pragma omp parallel for
            for (int i = 0; i < numElements; ++i)
            {
                output[i] -= input[i] * this->diagonalElements[i];
            }
        #endif
	}

	std::vector<T> getAbsRowSum()
	{
		std::vector<T> result(this->getNumRows());

		#pragma omp parallel for
		for (int k = 0; k < this->getNumRows(); ++k)
		{
			result[k] = std::abs(this->diagonalElements[k]);
		}

		return result;
	}

	T getMaxRowSumAbs()
	{
		Tdata diagonalElementsCopy = this->diagonalElements;

		vectorAbs(diagonalElementsCopy);

		return vectorMax(diagonalElementsCopy);
	}

	//transposing the identity does nothing
	void transpose(){}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA()
	{
		Tdata diagonalElementsCopy = this->diagonalElements;

		vectorAbs(diagonalElementsCopy);

		return diagonalElementsCopy;
	}
	#endif
};

#endif
