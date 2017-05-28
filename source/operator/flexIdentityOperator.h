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
		int numRows = this->getNumRows();
		int numCols = this->getNumCols();

		if (this->isMinus)
		{
			if(transposed)
			{
				#pragma omp parallel for
				for (int i = 0; i < numCols; ++i)
				{
					if(numCols > numRows && i >= numRows)
						output[i] = 0;
					else
						output[i] = -input[i];
				}
			}
			else
			{
				#pragma omp parallel for
				for (int i = 0; i < numRows; ++i)
				{
					if(numRows > numCols && i >= numCols)
						output[i] = 0;
					else
						output[i] = -input[i];
				}
			}
		}
		else
		{
			if(transposed)
			{
				#pragma omp parallel for
				for (int i = 0; i < numCols; ++i)
				{
					if(numCols > numRows && i >= numRows)
						output[i] = 0;
					else
						output[i] = input[i];
				}
			}
			else
			{
				#pragma omp parallel for
				for (int i = 0; i < numRows; ++i)
				{
					if(numRows > numCols && i >= numCols)
						output[i] = 0;
					else
						output[i] = input[i];
				}
			}
		}
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		if (this->isMinus)
		{
			doTimesMinus(transposed, input, output);
		}
		else
		{
			doTimesPlus(transposed, input, output);
		}
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		if (this->isMinus)
		{
			doTimesPlus(transposed, input, output);
		}
		else
		{
			doTimesMinus(transposed, input, output);
		}
	}

	T getMaxRowSumAbs(bool transposed)
	{
		return static_cast<T>(1);
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		std::vector<T> result;

		if(transposed)
			result = std::vector<T>(this->getNumCols(), (T)1);
		else
			result = std::vector<T>(this->getNumRows(), (T)1);

		return result;
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		thrust::device_vector<T> result;

		if(transposed)
			result = thrust::device_vector<T>(this->getNumCols(), (T)1);
		else
			result = thrust::device_vector<T>(this->getNumRows(), (T)1);

		return result;
	}
	#endif

private:
	void doTimesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		int numRows = this->getNumRows();
		int numCols = this->getNumCols();

		if(transposed)
		{
			#ifdef __CUDACC__
				if(numCols <= numRows)
					thrust::transform(output.begin(), output.end(), input.begin(), output.begin(), thrust::plus<T>());
				else
				{
					thrust::transform(output.begin(), output.begin() + numRows, input.begin(), output.begin(), thrust::plus<T>());
				}

			#else
				#pragma omp parallel for
				for (int i = 0; i < numCols; ++i)
				{
					if(numCols <= numRows || i < numRows)
						output[i] += input[i];
				}
			#endif
		}
		else
		{
			#ifdef __CUDACC__
				if(numRows <= numCols)
					thrust::transform(output.begin(), output.end(), input.begin(), output.begin(), thrust::plus<T>());
				else
				{
					thrust::transform(output.begin(), output.begin() + numCols, input.begin(), output.begin(), thrust::plus<T>());
				}
			#else
				#pragma omp parallel for
				for (int i = 0; i < numRows; ++i)
				{
					if(numRows <= numCols || i < numCols)
						output[i] += input[i];
				}
			#endif
		}
	}

	void doTimesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		int numRows = this->getNumRows();
		int numCols = this->getNumCols();

		if(transposed)
		{
			#ifdef __CUDACC__
				if(numCols <= numRows)
					thrust::transform(output.begin(), output.end(), input.begin(), output.begin(), thrust::minus<T>());
				else
				{
					thrust::transform(output.begin(), output.begin() + numRows, input.begin(), output.begin(), thrust::minus<T>());
				}

			#else
				#pragma omp parallel for
				for (int i = 0; i < numCols; ++i)
				{
					if(numCols <= numRows || i < numRows)
						output[i] -= input[i];
				}
			#endif
		}
		else
		{
			#ifdef __CUDACC__
				if(numRows <= numCols)
					thrust::transform(output.begin(), output.end(), input.begin(), output.begin(), thrust::minus<T>());
				else
				{
					thrust::transform(output.begin(), output.begin() + numCols, input.begin(), output.begin(), thrust::minus<T>());
				}
			#else
				#pragma omp parallel for
				for (int i = 0; i < numRows; ++i)
				{
					if(numRows <= numCols || i < numCols)
						output[i] -= input[i];
				}
			#endif
		}
	}
};

#endif
