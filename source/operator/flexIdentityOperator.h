#ifndef flexIdentityOperator_H
#define flexIdentityOperator_H


#include "vector"
#include "tools.h"
#include "flexLinearOperator.h"

//! represents an identiy operator
template<typename T>
class flexIdentityOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:

	//! initializes the identiy operator
	/*!
		\param aNumRows number of rows
		\param aNumCols number of cols
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexIdentityOperator(int aNumRows, int aNumCols, bool aMinus) : flexLinearOperator<T>(aNumRows, aNumCols, identityOp, aMinus) {}

	flexIdentityOperator<T>* copy()
	{
		flexIdentityOperator<T>* A = new flexIdentityOperator<T>(this->getNumRows(), this->getNumCols(), this->isMinus);

		return A;
	}

	//apply linear operator to vector
	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		int numElements = (int)output.size();

		if (this->isMinus)
		{
			int numElements = (int)input.size();
			#pragma omp parallel for
			for (int i = 0; i < numElements; ++i)
			{
				output[i] = -input[i];
			}
		}
		else
		{
			int numElements = (int)input.size();
			#pragma omp parallel for
			for (int i = 0; i < numElements; ++i)
			{
				output[i] = input[i];
			}
		}
	}

		void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		if (this->isMinus)
		{
			doTimesMinus(input, output);
		}
		else
		{
			doTimesPlus(input, output);
		}
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		if (this->isMinus)
		{
			doTimesPlus(input, output);
		}
		else
		{
			doTimesMinus(input, output);
		}
	}

	T getMaxRowSumAbs(bool transposed)
	{
		return static_cast<T>(1);
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		std::vector<T> result(this->getNumRows(), (T)1);

		return result;
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		thrust::device_vector<T> result(this->getNumRows(), (T)1);

		return result;
	}
	#endif

private:
	void doTimesPlus(const Tdata &input, Tdata &output)
	{
		#ifdef __CUDACC__
			thrust::transform(output.begin(), output.end(), input.begin(), output.begin(), thrust::plus<T>());
		#else
            int numElements = (int)input.size();
            #pragma omp parallel for
            for (int i = 0; i < numElements; ++i)
            {
                output[i] += input[i];
            }
		#endif
	}

	void doTimesMinus(const Tdata &input, Tdata &output)
	{
		#ifdef __CUDACC__
			thrust::transform(output.begin(), output.end(), input.begin(), output.begin(), thrust::minus<T>());
		#else
            int numElements = (int)input.size();
            #pragma omp parallel for
            for (int i = 0; i < numElements; ++i)
            {
                output[i] -= input[i];
            }
		#endif
	}
};

#endif
