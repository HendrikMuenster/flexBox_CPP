#ifndef flexMatrix_H
#define flexMatrix_H

#include "flexLinearOperator.h"

#include <vector>

template < typename T, typename Tvector >
class flexMatrix : public flexLinearOperator<T, Tvector>
{
private:
	std::vector<int> rowToIndexList;
	std::vector<int> indexList;
	Tvector valueList;

public:
	flexMatrix(void) : indexList(), valueList(), rowToIndexList(), flexLinearOperator<T, Tvector>(0, 0, matrixGPUOp){};

	flexMatrix(int  _numRows, int  _numCols) : rowToIndexList(_numRows + 1, static_cast<int>(0)), indexList(0, 0), valueList(0, 0), flexLinearOperator<T, Tvector>(_numRows, _numCols, matrixGPUOp){};

	flexMatrix<T, Tvector>* copy()
	{
		flexMatrix<T, Tvector>* A = new flexMatrix<T, Tvector>(this->getNumRows(), this->getNumCols());
		
		A->rowToIndexList = rowToIndexList;
		A->indexList = indexList;
		A->valueList = valueList;

		return A;
	}


	//todo
	void times(bool transposed, const Tvector &input, Tvector &output)
	{

	}

	void timesPlus(bool transposed, const Tvector &input, Tvector &output)
	{
		if (transposed)
		{
			//todo: check if transposed multiplication can be parallelized
			for (int i = 0; i < this->getNumRows(); ++i)
			{
				T rowsum = (T)0;
				// initialize result
				int indexNext = rowToIndexList[i + 1];
				for (int index = rowToIndexList[i]; index < indexNext; ++index)
				{
					output[indexList[index]] += input[i] * valueList[index];
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
					rowsum += input[indexList[index]] * valueList[index];
				}
				output[i] += rowsum;
			}
		}
	}

	void timesMinus(bool transposed, const Tvector &input, Tvector &output)
	{
		if (transposed)
		{
			//todo: check if transposed multiplication can be parallelized
			for (int i = 0; i < this->getNumRows(); ++i)
			{
				T rowsum = (T)0;
				// initialize result
				int indexNext = rowToIndexList[i + 1];
				for (int index = rowToIndexList[i]; index < indexNext; ++index)
				{
					output[indexList[index]] -= input[i] * valueList[index];
				}
				
			}
		}
		else
		{
			#pragma omp parallel for
			for (int i = 0; i < this->getNumRows(); i++)
			{
				T rowsum = (T)0;
				// initialize result
				int indexNext = rowToIndexList[i + 1];
				for (int index = rowToIndexList[i]; index < indexNext; ++index)
				{
					rowsum += input[indexList[index]] * valueList[index];
				}
				output[i] -= rowsum;
			}
		}
	}

	//this is the fast way to fill flexMatrix
	void blockInsert(std::vector<int> &indexI,const  std::vector<int> &indexJ,const Tvector &indexVal)
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
			int bucketIndex = indexI[indexInput];
			buckets[bucketIndex].push_back(indexInput);
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
				valueList.push_back(indexVal[tmpIndex]);
				++numElements;
			}

			//update rowToIndexList
			rowToIndexList[indexRow + 1] = rowToIndexList[indexRow] + numElements;
		}

	}

	/*
	//inserts new matrix element val at position [i][j]. This is SLOW!
	void insertElement(int i, int j, T val)
	{
		//get start position of next row
		int startIndexNextRow = rowToIndexList[i + 1];

		int numElt = indexList.size();

		//increment size of index and value list by 1
		indexList.push_back(0);
		valueList.push_back(static_cast<T>(0));
		//indexList.resize(indexList.size() + 1,static_cast<T>(0));
		//valueList.resize(valueList.size() + 1,static_cast<T>(0));

		//shift all elements starting with startIndexNextRow to next position
		for (int index = indexList.size()-1; index > startIndexNextRow; index--)
		{
			indexList[index] = indexList[index - 1];
			valueList[index] = valueList[index - 1];
		}

		//update indexList and valueList at current position
		indexList[startIndexNextRow] = j;
		valueList[startIndexNextRow] = val;

		//increase all elemets above i in rowToIndexList
		for (int index = i + 1; index < numRows+1; index++)
		{
			++rowToIndexList[index];
		}
	}*/

	//todo: remove or change and incorporate transposed
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
					result[indexList[index]] += std::abs(valueList[index]);
				}
			}
			
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
					tmpSum += std::abs(valueList[index]);
				}


				result[k] = tmpSum;
			}
			
			return result;
		}
	}


	void printRow(int i)
	{
		for (int index = rowToIndexList[i]; index < rowToIndexList[i+1]; ++index)
		{
			printf("(%d,%d,%f)|", i, indexList[index], valueList[index]);
		}

		printf("\n");

	}
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
};

#endif