#ifndef flexFullMatrix_H
#define flexFullMatrix_H

#include "flexLinearOperator.h"

#include <vector>

//! represents a full (non-CUDA) matrix
template<typename T>
class flexFullMatrix : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	Tdata valueList;

public:
	//! initializes an empty matrix
	flexFullMatrix() : valueList(), flexLinearOperator<T>(0, 0, matrixOp, false) {};

	//! initializes a matrix
	/*!
		\param aNumRows number of rows
		\param aNumCols number of cols
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexFullMatrix(int aNumRows, int aNumCols, bool aMinus) : valueList(aNumRows*aNumCols, 0), flexLinearOperator<T>(aNumRows, aNumCols, matrixOp, aMinus){};

	flexFullMatrix<T>* copy()
	{
		flexFullMatrix<T>* A = new flexFullMatrix<T>(this->getNumRows(), this->getNumCols(), this->isMinus);

		A->valueList = valueList;

		return A;
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{

	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		if (this->isMinus)
		{
			doTimesCPU(transposed, input, output,MINUS);
		}
		else
		{
			doTimesCPU(transposed, input, output,PLUS);
		}
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		if (this->isMinus)
		{
			doTimesCPU(transposed, input, output,PLUS);
		}
		else
		{
			doTimesCPU(transposed, input, output,MINUS);
		}
	}

	//inserts new matrix element val at position [i][j]
	void insertElement(int i, int j, T val)
	{
        this->valueList[index2DtoLinear(i,j)] = val;
	}

	void insertElement(int i, T val)
	{
		this->valueList[i] = val;
	}
    
    int index2DtoLinear(int i, int j)
	{
		return i + j*this->getNumRows();
	}

	T getMaxRowSumAbs(bool transposed)
	{
		std::vector<T> rowSum = this->getAbsRowSum(transposed);

		return *std::max_element(rowSum.begin(), rowSum.end());
	}


	std::vector<T> getAbsRowSum(bool transposed)
	{
		if (transposed)
		{
			std::vector<T> result(this->getNumCols(), (T)0);

			for (int i = 0; i < this->getNumRows(); ++i)
			{
				for (int j = 0; j < this->getNumCols(); ++j)
				{
					result[j] += std::abs(valueList[index2DtoLinear(i,j)]);
				}
			}

			/*for (int i = 0; i < this->getNumCols(); ++i)
			{
				printf("T %f\n", result[i]);
			}*/

			return result;
		}
		else
		{
			std::vector<T> result(this->getNumRows(),(T)0);

			for (int i = 0; i < this->getNumRows(); ++i)
			{
				for (int j = 0; j < this->getNumCols(); ++j)
				{
					result[i] += std::abs(valueList[index2DtoLinear(i, j)]);
				}
			}

			/*for (int i = 0; i < this->getNumRows(); ++i)
			{
				printf(" %f\n", result[i]);
			}*/

			return result;
		}
	}

	//! prints requested row
	/*!
		\param i row to be printed
	*/
	void printRow(int i)
	{
		for (int j = 0; j < this->getNumCols(); ++j)
		{
			printf("(%d,%d,%f)|", i, j, valueList[index2DtoLinear(i, j)]);
		}

		printf("\n");
	}

	//! prints the whole matrix
	void printMatrix()
	{
		for (int i = 0; i < this->getNumRows(); i++)
		{
			printRow(i);
		}
	}

    //DUMMY FUNCTION
    #ifdef __CUDACC__
    thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		thrust::device_vector<T> result(this->getNumRows(), (T)1);

		return result;
	}
    #endif

	private:
	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output,const mySign s)
	{
        if (transposed)
		{
			#pragma omp parallel for
			for (int j = 0; j < this->getNumCols(); ++j)
			{
				T tmp = static_cast<T>(0); 
				
				for (int i = 0; i < this->getNumRows(); ++i)
				{
					tmp += input[i] * valueList[index2DtoLinear(i, j)];
				}

				switch (s)
				{
					case PLUS:
					{
						output[j] += tmp;
						break;
					}
					case MINUS:
					{
						output[j] -= tmp;
						break;
					}
				}
			}
        }
        else
        {
			for (int j = 0; j < this->getNumCols(); ++j)
			{
				T tmp = input[j];
                #pragma omp parallel for
				for (int i = 0; i < this->getNumRows(); ++i)
				{
					switch (s)
					{
						case PLUS:
						{
							output[i] += tmp * valueList[index2DtoLinear(i, j)];
							break;
						}
						case MINUS:
						{
							output[i] -= tmp * valueList[index2DtoLinear(i, j)];
							break;
						}
					}
				}
			}
        }
    }
};

#endif
