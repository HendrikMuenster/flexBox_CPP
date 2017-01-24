#ifndef flexIdentityOperator_H
#define flexIdentityOperator_H


#include "vector"
#include "tools.h"
#include "flexLinearOperator.h"

template < typename T, typename Tvector >
class flexIdentityOperator : public flexLinearOperator<T, Tvector>
{
private:
	bool minus;
public:

	flexIdentityOperator(int _numRows, int _numCols, bool _minus) : flexLinearOperator<T, Tvector>(_numRows, _numCols, identityOp, _minus)
	{
		minus = _minus;
	};

	flexIdentityOperator<T, Tvector>* copy()
	{
		flexIdentityOperator<T, Tvector>* A = new flexIdentityOperator<T, Tvector>(this->getNumRows(), this->getNumCols(), this->minus);

		return A;
	}

	//apply linear operator to vector
	void times(bool transposed, const Tvector &input, Tvector &output)
	{
		int numElements = (int)output.size();

		if (this->minus == true)
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

	void doTimesPlus(const Tvector &input, Tvector &output)
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

	void doTimesMinus(const Tvector &input, Tvector &output)
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

	void timesPlus(bool transposed, const Tvector &input, Tvector &output)
	{
		if (this->minus == true)
		{
			doTimesMinus(input, output);
		}
		else
		{
			doTimesPlus(input, output);
		}
	}

	void timesMinus(bool transposed, const Tvector &input, Tvector &output)
	{
		if (this->minus == true)
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
};

#endif
