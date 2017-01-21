#ifndef flexZeroOperator_H
#define flexZeroOperator_H

#include <vector>
#include "flexLinearOperator.h"

template<typename T>
class flexZeroOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:
	flexZeroOperator(int aNumRows, int aNumCols) : flexLinearOperator<T>(aNumRows, aNumCols, zeroOp){};

	flexZeroOperator<T>* copy()
	{
		flexZeroOperator<T>* A = new flexZeroOperator<T>(this->getNumRows(), this->getNumCols());

		return A;
	}


	//apply linear operator to vector
	void times(const Tdata &input, Tdata &output)
	{
		vectorScalarSet(output, (T)0);
	}

	void timesPlus(const Tdata &input, Tdata &output){}

	void timesMinus(const Tdata &input, Tdata &output){}

	T getMaxRowSumAbs()
	{
		return static_cast<T>(1);
	}

	std::vector<T> getAbsRowSum()
	{
		std::vector<T> result(this->getNumRows(),(T)0);

		return result;
	}

	//transposing the identity does nothing
	void transpose()
	{
		int numRowsTmp = this->getNumRows();
		this->setNumRows(this->getNumCols());
		this->setNumCols(numRowsTmp);
	}

#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA()
	{
		Tdata result(this->getNumRows(),(T)0);

		return result;
	}
#endif
};

#endif
