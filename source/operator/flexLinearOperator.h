#ifndef flexLinearOperator_H
#define flexLinearOperator_H

#include <vector>
#include "tools.h"

//! abstract base class for linear operators
/*!
	flexLinearOperator combines the interface for all usable operators
*/
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
	linOp type; 	//!< type of linear operator \sa linOp
	bool isMinus;	//!< determines if operator is negated

	//! initializes the linear operator
	/*!
		\param aNumRows number of rows of liner operator
		\param aNumCols number of columns of liner operator
		\param aType type of linear operator sa type \sa type
		\param aIsMinus determines if operator is negated \sa isMinus
	*/
	flexLinearOperator(int aNumRows, int aNumCols, linOp aType, bool aIsMinus) : numRows(aNumRows), numCols(aNumCols), type(linearOp), isMinus(aIsMinus)
	{

	}

	virtual ~flexLinearOperator()
	{
		if (VERBOSE > 0) printf("Linear operator destructor");
	}

	//! returns number of columns of the linear operator
	/*!
		\return number of columns
	*/
	int getNumCols() const
	{
		return numCols;
	}

	//! returns number of rows of the linear operator
	/*!
		\return number of rows
	*/
	int getNumRows() const
	{
		return numRows;
	}

	//! sets the number of columns of the linear operator
	/*!
		\param aNumCols number of columns
	*/
	void setNumCols(int aNumCols)
	{
		numCols = aNumCols;
	}

	//! sets the number of rows of the linear operator
	/*!
		\param aNumRows number of rows
	*/
	void setNumRows(int aNumRows)
	{
		numRows = aNumRows;
	}

	//! constrols if operator should be negated or not
	/*!
		\param aIsMinus true if operator should be negated, otherwise false.
	*/
	void setMinus(bool aIsMinus)
  {
      this->isMinus = aIsMinus;
  }

	//! copies the linear operator
	/*!
		\return copy of linear operator
	*/
	virtual flexLinearOperator<T>* copy() = 0;

	//! applies linear operator on vector
	/*!
		equals \f$ y = Ax \f$
		\param transposed is true if operator should be (temporarily) transposed before usage
		\param input data to be processed
		\param output output data
	*/
	virtual void times(bool transposed, const Tdata &input, Tdata &output) = 0;

	//! applies linear operator on vector and adds its result to y
	/*!
		equals \f$ y = y + Ax \f$
		\param transposed is true if operator should be (temporarily) transposed before usage
		\param input data to be processed
		\param output output data
	*/
	virtual void timesPlus(bool transposed, const Tdata &input, Tdata &output) = 0;

	//! applies linear operator on vector and substracts its result from y
	/*!
		equals \f$ y = y - Ax \f$
		\param transposed is true if operator should be (temporarily) transposed before usage
		\param input data to be processed
		\param output output data
	*/
	virtual void timesMinus(bool transposed, const Tdata &input, Tdata &output) = 0;

	//! returns a vector of sum of absolute values per row used for preconditioning
	/*!
		\param transposed is true if operator should be (temporarily) transposed before usage
		\return vector of sum of absolute values per row
	*/
	virtual std::vector<T> getAbsRowSum(bool transposed) = 0;

	#ifdef __CUDACC__
	//! same function as getAbsRowSum() but implemented in CUDA
	/*!
		\param transposed is true if operator should be (temporarily) transposed before usage
		\return vector of sum of absolute values per row
	*/
		virtual thrust::device_vector<T> getAbsRowSumCUDA(bool transposed) = 0;
	#endif

	//! returns the maximum sum of absolute values per row used for preconditioning
	/*!
		\param transposed is true if operator should be (temporarily) transposed before usage
		\return maximum sum of absolute values per row
	*/
	virtual T getMaxRowSumAbs(bool transposed) = 0;
};

#endif
