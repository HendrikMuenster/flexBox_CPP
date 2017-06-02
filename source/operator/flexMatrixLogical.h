#ifndef flexMatrixLogical_H
#define flexMatrixLogical_H

#include "flexLinearOperator.h"

#include <vector>

//! represents a full (non-CUDA) logical matrix
template<typename T>
class flexMatrixLogical : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:


public:
	std::vector<int> rowToIndexList;
	std::vector<int> indexList;//
	//! initializes an empty matrix
	flexMatrixLogical() : rowToIndexList(), indexList(), flexLinearOperator<T>(0, 0, matrixOp, false) {};

	//! initializes a matrix
	/*!
		\param aNumRows number of rows
		\param aNumCols number of cols
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexMatrixLogical(int aNumRows, int aNumCols, bool aMinus) : rowToIndexList(aNumRows + 1, static_cast<int>(0)), indexList(0, 0), flexLinearOperator<T>(aNumRows, aNumCols, matrixOp, aMinus){};

	flexMatrixLogical<T>* copy()
	{
		flexMatrixLogical<T>* A = new flexMatrixLogical<T>(this->getNumRows(), this->getNumCols(), this->isMinus);

		A->rowToIndexList = indexList;
		A->indexList = rowToIndexList;

		return A;
	}

	//todo
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

	//! inserts position of all non-zero elements into matrix
	/*!
	this is the fastest way to fill flexMatrixLogical
	\param indexI vector of row indices
	\param indexJ vector of column indices
	*/
	void blockInsert(const std::vector<int> &indexI, const std::vector<int> &indexJ)
	{
		//clear matrix
		//clear();

		int numberListElements = (int)indexI.size();

		//initialize vecvector
		std::vector<int> emptyBucket(0, 0);
		std::vector < std::vector<int> > buckets(this->getNumRows(), emptyBucket);

		//add elements to buckets
		for (int indexInput = 0; indexInput < numberListElements; indexInput++)
		{
			buckets[indexI[indexInput]].push_back(indexInput);
		}

		//go trough all rows:
		for (int indexRow = 0; indexRow < this->getNumRows(); indexRow++)
		{
			int numElements = 0;

			//go through bucket
			for (int indexBucket = 0; indexBucket < (int)buckets[indexRow].size(); indexBucket++)
			{
				int tmpIndex = buckets[indexRow][indexBucket];

				indexList.push_back(indexJ[tmpIndex]);
				++numElements;
			}

			//update rowToIndexList
			rowToIndexList[indexRow + 1] = rowToIndexList[indexRow] + numElements;
		}
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
			std::vector<T> result(this->getNumCols());

			//todo check if omp is possible
			for (int k = 0; k < this->getNumRows(); ++k)
			{
				for (int index = rowToIndexList[k]; index < rowToIndexList[k + 1]; ++index)
				{
					result[indexList[index]] += (T)1;
				}
			}

			/*for (int k = 0; k < this->getNumCols(); ++k)
			{
				printf("T%f\n", result[k]);
			}*/

			return result;
		}
		else
		{
			std::vector<T> result(this->getNumRows());

			#pragma omp parallel for
			for (int k = 0; k < this->getNumRows(); ++k)
			{
				T tmpSum = static_cast<T>(0);
				for (int index = rowToIndexList[k]; index < rowToIndexList[k + 1]; ++index)
				{
					tmpSum += (T)1;
				}

				result[k] = tmpSum;
			}

			/*for (int k = 0; k < this->getNumRows(); ++k)
			{
				printf("%f\n", result[k]);
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
		for (int index = rowToIndexList[i]; index < rowToIndexList[i + 1]; ++index)
		{
			printf("(%d,%d,1)|", i, indexList[index]);
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
		void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, const mySign s)
		{
			if (transposed)
			{
				//todo: check if transposed multiplication can be parallelized
				#pragma omp parallel for
				for (int i = 0; i < this->getNumRows(); ++i)
				{
					int indexNext = rowToIndexList[i + 1];
					for (int index = rowToIndexList[i]; index < indexNext; ++index)
					{
						switch (s)
						{
							case PLUS:
							{
								output[indexList[index]] += input[i];
								break;
							}
							case MINUS:
							{
								output[indexList[index]] -= input[i];
								break;
							}
						}
					}
				}
			}
			else
			{
				#pragma omp parallel for
				for (int i = 0; i < this->getNumRows(); ++i)
				{
					T rowsum = (T)0;
					// initialize result
					int indexNext = rowToIndexList[i + 1];
					for (int index = rowToIndexList[i]; index < indexNext; ++index)
					{
						rowsum += input[indexList[index]];
					}

					switch (s)
					{
						case PLUS:
						{
							output[i] += rowsum;
							break;
						}
						case MINUS:
						{
							output[i] -= rowsum;
							break;
						}
					}
				}
			}
		}
};

#endif
