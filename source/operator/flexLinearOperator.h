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

	flexLinearOperator(int aNumRows, int aNumCols) : numRows(aNumRows), numCols(aNumCols), type(linearOp)
	{

	}

	virtual ~flexLinearOperator()
	{
		if (VERBOSE > 0) printf("Linear operator destructor");
	}

	flexLinearOperator(int aNumRows, int aNumCols, linOp aType) : numCols(aNumCols), type(aType)
	{

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

	virtual flexLinearOperator<T>* copy() = 0;

	//apply linear operator to vector
	virtual void times(const Tdata &input, Tdata &output) = 0;

	virtual void timesPlus(const Tdata &input, Tdata &output) = 0;

	virtual void timesMinus(const Tdata &input, Tdata &output) = 0;

	virtual std::vector<T> getAbsRowSum() = 0;

	#ifdef __CUDACC__
		virtual thrust::device_vector<T> getAbsRowSumCUDA() = 0;
	#endif

	//used for preconditioning
	virtual T getMaxRowSumAbs() = 0;

	//transpose current matrix
	virtual void transpose() = 0;
};

#endif
