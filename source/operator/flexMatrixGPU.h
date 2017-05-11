#ifndef flexMatrixGPU_H
#define flexMatrixGPU_H

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <vector>

#include "flexLinearOperator.h"

//! represents a (CUDA) matrix
template<typename T>
class flexMatrixGPU : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	cusparseHandle_t handle;
	cusparseMatDescr_t descrA;

	int* listRowEntries;
	int* listColIndices;
	T* listValues;

	int nnz;

public:
	//! initializes a matrix
	/*!
		\param aNumRows number of rows
		\param aNumCols number of cols
		\param rowList array of row indices
		\param colList array of column indices
		\param indexVal array of data corresponding row and column indices
		\param formatCRS determines if the input data is in compressed row storage
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexMatrixGPU(int  aNumRows, int  aNumCols, int* rowList, int* colList, T* indexVal, bool formatCRS, bool aMinus) : flexLinearOperator<T>(aNumRows, aNumCols, matrixGPUOp, aMinus)
	{
		//create sparse matrix
		cusparseCreate(&this->handle);
		cusparseCreateMatDescr(&this->descrA);

		cusparseSetMatType(this->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(this->descrA, CUSPARSE_INDEX_BASE_ZERO);

		//if formatCRS is true then the input data is already in compressed row storage format, otherwise we have to convert it
		if (formatCRS)
		{
			this->nnz = rowList[aNumRows]; //access last entry
		}
		else
		{
			this->nnz = colList[aNumCols]; //access last entry
		}

		cudaMalloc(&this->listValues, this->nnz * sizeof(T));
		cudaMalloc(&this->listColIndices, this->nnz * sizeof(int));
		cudaMalloc(&this->listRowEntries, (aNumRows + 1) * sizeof(int));

		if (formatCRS == false)
		{
			//copy input to device memory
			T* listValuesTmp;
			int* listColIndicesTmp;
			int* listRowEntriesTmp;

			cudaMalloc(&listValuesTmp, this->nnz * sizeof(T));
			cudaMemcpy(listValuesTmp, indexVal, this->nnz * sizeof(T), cudaMemcpyHostToDevice);
			cudaMalloc(&listColIndicesTmp, (aNumCols + 1) * sizeof(int));
			cudaMemcpy(listColIndicesTmp, colList, (aNumCols + 1) * sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc(&listRowEntriesTmp, this->nnz  * sizeof(int));
			cudaMemcpy(listRowEntriesTmp, rowList, this->nnz * sizeof(int), cudaMemcpyHostToDevice);


			cudaDeviceSynchronize();
			cusparseStatus_t status = cusparseScsr2csc(this->handle, aNumCols, aNumRows, this->nnz, listValuesTmp, listColIndicesTmp, listRowEntriesTmp, this->listValues, this->listColIndices, this->listRowEntries, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
			cudaDeviceSynchronize();

			cudaFree(listValuesTmp);
			cudaFree(listColIndicesTmp);
			cudaFree(listRowEntriesTmp);

			if (VERBOSE > 0)
			{
				switch (status)
				{
				case CUSPARSE_STATUS_SUCCESS:
				{
					printf("Copy was successfull\n");
					break;
				}
				case CUSPARSE_STATUS_NOT_INITIALIZED:
				{
					printf("the library was not initialized\n");
					break;
				}
				case CUSPARSE_STATUS_ALLOC_FAILED:
				{
					printf("the resources could not be allocated\n");
					break;
				}
				case CUSPARSE_STATUS_INVALID_VALUE:
				{
					printf("invalid parameters were passed(m, n, nnz<0)\n");
					break;
				}
				case CUSPARSE_STATUS_ARCH_MISMATCH:
				{
					printf("the device does not support double precision\n");
					break;
				}
				case CUSPARSE_STATUS_EXECUTION_FAILED:
				{
					printf("the function failed to launch on the GPU\n");
					break;
				}
				case CUSPARSE_STATUS_INTERNAL_ERROR:
				{
					printf("the function failed to launch on the GPU\n");
					break;
				}
				default:
				{
					printf("Error Copy!");
					break;
				}
				}
			}
		}
		else
		{
			cudaMemcpy(this->listValues, indexVal, this->nnz * sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(this->listColIndices, colList, this->nnz * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(this->listRowEntries, rowList, (aNumRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
		}
	};

	~flexMatrixGPU()
	{
		if (VERBOSE > 0) printf("MatrixGPU destructor!");
		//free cuda memory
		cudaFree(this->listValues);
		cudaFree(this->listColIndices);
		cudaFree(this->listRowEntries);
	}

	flexMatrixGPU<T>* copy()
	{
		//copy matrix data to host

		//allocate memory
		T *hostValues = (T *)malloc(this->nnz * sizeof(T));
		int *hostRowIndices = (int *)malloc((this->getNumRows() + 1) * sizeof(int));
		int *hostColIndices = (int *)malloc(this->nnz * sizeof(int));

		cudaMemcpy(hostValues, this->listValues, this->nnz * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostRowIndices, this->listRowEntries, (this->getNumRows() + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostColIndices, this->listColIndices, this->nnz * sizeof(int), cudaMemcpyDeviceToHost);

		flexMatrixGPU<T>* A = new flexMatrixGPU<T>(this->getNumRows(), this->getNumCols(), hostRowIndices, hostColIndices, hostValues,true, this->isMinus);

		free(hostValues);
		free(hostRowIndices);
		free(hostColIndices);

		return A;
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		T alpha;

        if (this->isMinus)
        {
           alpha = (T)-1;
        }
        else
        {
            alpha = (T)1;
        }

		const T beta = (T)0;

		T* ptrOutput = thrust::raw_pointer_cast(output.data());
		const T* ptrInput = thrust::raw_pointer_cast(input.data());

		if (transposed == false)
		{
			cusparseScsrmv(this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, this->getNumRows(), this->getNumCols(), nnz, &alpha, this->descrA, this->listValues, this->listRowEntries, this->listColIndices, ptrInput, &beta, ptrOutput);
		}
		else
		{
			cusparseScsrmv(this->handle, CUSPARSE_OPERATION_TRANSPOSE, this->getNumRows(), this->getNumCols(), nnz, &alpha, this->descrA, this->listValues, this->listRowEntries, this->listColIndices, ptrInput, &beta, ptrOutput);
		}
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		T alpha;

		if (this->isMinus)
        {
           alpha = (T)-1;
        }
        else
        {
           alpha = (T)1;
        }

		const T beta = (T)1;

		T* ptrOutput = thrust::raw_pointer_cast(output.data());
		const T* ptrInput = thrust::raw_pointer_cast(input.data());

		if (transposed == false)
		{
			cusparseScsrmv(this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, this->getNumRows(), this->getNumCols(), nnz, &alpha, this->descrA, this->listValues, this->listRowEntries, this->listColIndices, ptrInput, &beta, ptrOutput);
		}
		else
		{
			cusparseScsrmv(this->handle, CUSPARSE_OPERATION_TRANSPOSE, this->getNumRows(), this->getNumCols(), nnz, &alpha, this->descrA, this->listValues, this->listRowEntries, this->listColIndices, ptrInput, &beta, ptrOutput);
		}
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		T alpha;

		if (this->isMinus)
        {
          alpha = (T)1;
        }
        else
        {
            alpha = (T)-1;
        }
		const T beta = (T)1;

		T* ptrOutput = thrust::raw_pointer_cast(output.data());
		const T* ptrInput = thrust::raw_pointer_cast(input.data());

		if (transposed == false)
		{
			cusparseScsrmv(this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, this->getNumRows(), this->getNumCols(), nnz, &alpha, this->descrA, this->listValues, this->listRowEntries, this->listColIndices, ptrInput, &beta, ptrOutput);
		}
		else
		{
			cusparseScsrmv(this->handle, CUSPARSE_OPERATION_TRANSPOSE, this->getNumRows(), this->getNumCols(), nnz, &alpha, this->descrA, this->listValues, this->listRowEntries, this->listColIndices, ptrInput, &beta, ptrOutput);
		}
	}

	//TODO: implement. 1 is just placeholder
	T getMaxRowSumAbs(bool transposed)
	{
		return 1;
	}

  //dummy, this function is not used in a CUDA setting
	std::vector<T> getAbsRowSum(bool transposed)
	{
    std::vector<T> result(1);
		return result;
	}

	//! prints requested row
	/*!
		\param i row to be printed
	*/
	void printRow(int i)
	{

	}

	//! prints the whole matrix
	void printMatrix()
	{
		T *hostValues = (T *)malloc(this->nnz * sizeof(T));
		int *hostRowIndices = (int *)malloc((this->getNumRows() + 1) * sizeof(int));
		int *hostColIndices = (int *)malloc(this->nnz * sizeof(int));

		cudaMemcpy(hostValues, this->listValues, this->nnz * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostRowIndices, this->listRowEntries, (this->getNumRows() + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostColIndices, this->listColIndices, this->nnz * sizeof(int), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		free(hostValues);
		free(hostRowIndices);
		free(hostColIndices);
	}

	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
        std::vector<T> resultTmp;

        if (transposed == false)
        {
            resultTmp.resize(this->getNumRows());
        }
        else
        {
            resultTmp.resize(this->getNumCols());
        }

		//allocate memory
		T *hostValues = (T *)malloc(this->nnz * sizeof(T));
		int *hostRowIndices = (int *)malloc((this->getNumRows() + 1) * sizeof(int));
		int *hostColIndices = (int *)malloc(this->nnz * sizeof(int));

		cudaMemcpy(hostValues, this->listValues, this->nnz * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostRowIndices, this->listRowEntries, (this->getNumRows() + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostColIndices, this->listColIndices, this->nnz * sizeof(int), cudaMemcpyDeviceToHost);

		for (int row = 0; row < this->getNumRows(); row++)
        {
            int starting_col_index = hostRowIndices[row];
            int stopping_col_index = hostRowIndices[row + 1];
            if (starting_col_index == stopping_col_index)
                continue;
            else
            {
                for (int current_col_index = starting_col_index; current_col_index < stopping_col_index; current_col_index++)
                {
                    if (transposed == false)
                    {
                        resultTmp[row] += std::abs(hostValues[current_col_index]);
                    }
                    else
                    {
                        resultTmp[hostColIndices[current_col_index]] += std::abs(hostValues[current_col_index]);
                    }
                }
            }
        }

        free(hostValues);
		free(hostRowIndices);
		free(hostColIndices);

		Tdata result(resultTmp.size());

        thrust::copy(resultTmp.begin(), resultTmp.end(), result.begin());

		return result;
	}
};

#endif
