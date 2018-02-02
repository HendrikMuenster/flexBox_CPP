#ifndef flexSubsamplingOperator_H
#define flexSubsamplingOperator_H

#include <vector>
#include "tools.h"
#include "flexLinearOperator.h"
#include <algorithm>

//! represents an permutation operator. If you need repetition, consider using \sa flexSubsamplingOperator
template<typename T>
class flexSubsamplingOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif
private:
    std::vector<int> indices;
    std::vector<int> transIndices;

public:

	//! initializes the permutation operator
	/*!
		\param aNumRows number of rows
		\param aNumCols number of cols
        \param aIndices vector of indices, s.t. output(i) = input(aIndices(i)), every index must be in the vector exatly once
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexSubsamplingOperator(int aNumRows, int aNumCols, std::vector<int> aIndices,  bool aMinus) : flexLinearOperator<T>(aNumRows, aNumCols, permuteOp, aMinus)
        , indices(aIndices)
    {
        std::vector<int> transIndices(indices.size());
        std::iota(transIndices.begin(), transIndices.end(), 0);

        std::sort(transIndices.begin(), transIndices.end(), [&indices](int id1, int id2)
                {
                    return indices[id1] < indices[id2];
                });

    }

	flexSubsamplingOperator<T>* copy()
	{
		flexSubsamplingOperator<T>* A = new flexSubsamplingOperator<T>(this->getNumRows(), this->getNumCols(), this->isMinus);
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
                    output[i] = -input[transIndices[i]];
			}
			else
			{
				#pragma omp parallel for
				for (int i = 0; i < numRows; ++i)
                    output[i] = -input[indices[i]];
			}
		}
		else
		{
			if(transposed)
			{
				#pragma omp parallel for
				for (int i = 0; i < numCols; ++i)
                    output[i] = input[transIndices[i]];
			}
			else
			{
				#pragma omp parallel for
				for (int i = 0; i < numRows; ++i)
                    output[i] = input[indices[i]];
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

    //dummy
	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		thrust::device_vector<T> result;
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
            #pragma omp parallel for
            for (int i = 0; i < numCols; ++i)
                output[i] += input[transIndices[i]];
		}
		else
		{
            #pragma omp parallel for
            for (int i = 0; i < numRows; ++i)
                output[i] += input[indices[i]];
		}
	}

	void doTimesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		int numRows = this->getNumRows();
		int numCols = this->getNumCols();

		if(transposed)
		{
            #pragma omp parallel for
            for (int i = 0; i < numCols; ++i)
                output[i] -= input[transIndices[i]];
		}
		else
		{
            #pragma omp parallel for
            for (int i = 0; i < numRows; ++i)
                output[i] -= input[indices[i]];
		}
	}
};

#endif
