#ifndef flexLinearOperator_H
#define flexLinearOperator_H

#include "vector"
#include "tools.h"

template < typename T, typename Tvector >
class flexLinearOperator
{
private:
	int numRows;
	int numCols;
public:
	linOp type;
    bool isMinus;

	virtual ~flexLinearOperator()
	{
		if (VERBOSE > 0) printf("Linear operator destructor");
	}

	flexLinearOperator(int _numRows, int _numCols, linOp _type, bool _isMinus) : type(_type), isMinus(_isMinus)
	{
		numRows = _numRows;
		numCols = _numCols;
	}

	int getNumCols() const
	{
		return numCols;
	}

	int getNumRows() const
	{
		return numRows;
	}

	void setNumCols(int _numCols)
	{
		numCols = _numCols;
	}

	void setNumRows(int _numRows)
	{
		numRows = _numRows;
	}
    
    void setMinus(bool _isMinus)
    {
        this->isMinus = _isMinus;
    }

	virtual flexLinearOperator<T, Tvector>* copy() = 0;

	//apply linear operator to vector
	virtual void times(bool transposed, const Tvector &input, Tvector &output) = 0;

	virtual void timesPlus(bool transposed, const Tvector &input, Tvector &output) = 0;

	virtual void timesMinus(bool transposed, const Tvector &input, Tvector &output) = 0;

	virtual std::vector<T> getAbsRowSum(bool transposed) = 0;

	#ifdef __CUDACC__		
		virtual thrust::device_vector<T> getAbsRowSumCUDA(bool transposed) = 0;
	#endif

	//used for preconditioning
	virtual T getMaxRowSumAbs(bool transposed) = 0;
};

#endif
