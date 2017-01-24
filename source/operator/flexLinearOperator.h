#ifndef flexLinearOperator_H
#define flexLinearOperator_H

#include <vector>
#include "tools.h"

template <typename T>
class flexLinearOperator
{
	#ifdef __CUDACC__
		typedef thrust::device_vector<T> Tdata;
	#else
		typedef std::vector<T> Tdata;
	#endif

private:
	int numRows;
	int numCols;
public:
	linOp type;
	bool isMinus;

	flexLinearOperator(int aNumRows, int aNumCols, linOp _type, bool _isMinus) : numRows(aNumRows), numCols(aNumCols), type(linearOp), isMinus(_isMinus)
	{

	}

	virtual ~flexLinearOperator()
	{
		if (VERBOSE > 0) printf("Linear operator destructor");
	}

	int getNumCols() const
	{
		return numCols;
	}

	int getNumRows() const
	{
		return numRows;
	}

	void setNumCols(int aNumCols)
	{
		numCols = aNumCols;
	}

	void setNumRows(int aNumRows)
	{
		numRows = aNumRows;
	}
	
	void setMinus(bool _isMinus)
    {
        this->isMinus = _isMinus;
    }

	virtual flexLinearOperator<T>* copy() = 0;

	//apply linear operator to vector
	virtual void times(bool transposed, const Tdata &input, Tdata &output) = 0;

	virtual void timesPlus(bool transposed, const Tdata &input, Tdata &output) = 0;

	virtual void timesMinus(bool transposed, const Tdata &input, Tdata &output) = 0;

	virtual std::vector<T> getAbsRowSum(bool transposed) = 0;

	#ifdef __CUDACC__
		virtual thrust::device_vector<T> getAbsRowSumCUDA(bool transposed) = 0;
	#endif

	//used for preconditioning
	virtual T getMaxRowSumAbs(bool transposed) = 0;
};

#endif
