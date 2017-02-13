#ifndef flexProxDualLabeling_H
#define flexProxDualLabeling_H

#include "flexProx.h"

//put cuda kernel
#ifdef __CUDACC__
#define MAX_NUMBER_LABELS 16

template<typename T>
__global__ void dualLabelingProxCUDA(T** listYPtr, T** listYTildePtr, T** listFPtr, T** listSigmaPtr, int* dualNumbers, int numElements, int numDualVars)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= numElements)
		return;


	T tmpVector[MAX_NUMBER_LABELS]; 
	T tmpVector2[MAX_NUMBER_LABELS]; 

	//copy from yTilde
	for (int j = 0; j < numDualVars; j++)
	{
		tmpVector[j] = (listYTildePtr[dualNumbers[j]][i] - listFPtr[j][i]) / listSigmaPtr[dualNumbers[j]][i];
		tmpVector2[j] = tmpVector[j];
	}

	//sort tmpVector
	for (int c = 0 ; c < ( numDualVars - 1 ) ; c++ )
	{
		int position = c;

		for (int d = c + 1 ; d < numDualVars ; d++ )
		{
			if ( tmpVector[position] > tmpVector[d] )
			{
				position = d;
			}
		}
		if (position != c)
		{
			T swap = tmpVector[c];
			tmpVector[c] = tmpVector[position];
			tmpVector[position] = swap;
		}
   }

	T sumY = (T)0;
	T tOpt;

	int j = numDualVars - 2;
	while (j >= 0)
	{
		sumY = sumY + tmpVector[j + 1];

		tOpt = (sumY - 1) / (numDualVars - (j + 1));

		if (tOpt >= tmpVector[j])
		{
			break;
		}
		else
		{
			j = j - 1;
		}
	}

	if (j < 0)
	{
		tOpt = (sumY + tmpVector[0] - 1) / numDualVars;
	}

	//write result
	for (int j = 0; j < numDualVars; j++)
	{
		listYPtr[dualNumbers[j]][i] = listYTildePtr[dualNumbers[j]][i] - listSigmaPtr[dualNumbers[j]][i] * max(tmpVector2[j] - tOpt, (T)0);
	}
}

#endif


template<typename T>
class flexProxDualLabeling : public flexProx<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:

    flexProxDualLabeling() : flexProx<T>(dualL1AnisoProx){}

	~flexProxDualLabeling()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}
	

	
	

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{
		#ifdef __CUDACC__
			int numElements = (int)data->yTilde[dualNumbers[0]].size();
			int numDualVars = (int)dualNumbers.size();

			thrust::device_vector<T*> listYPtr(numDualVars);
			thrust::device_vector<T*> listYTildePtr(numDualVars);
			thrust::device_vector<T*> listFPtr(numDualVars);
			thrust::device_vector<T*> listSigmaPtr(numDualVars);

			thrust::device_vector<int> dualNumbersCUDA(dualNumbers);
				
			for (int j = 0; j < numDualVars; j++)
			{
				listYPtr[j] = thrust::raw_pointer_cast(data->y[dualNumbers[j]].data());
				listYTildePtr[j] = thrust::raw_pointer_cast(data->yTilde[dualNumbers[j]].data());
				listFPtr[j] = thrust::raw_pointer_cast(fList[j].data());
				listSigmaPtr[j] = thrust::raw_pointer_cast(data->sigmaElt[dualNumbers[j]].data());
			}

			T** YPtr = thrust::raw_pointer_cast(listYPtr.data());
			T** YTildePtr = thrust::raw_pointer_cast(listYTildePtr.data());
			T** FPtr = thrust::raw_pointer_cast(listFPtr.data());
			T** SigmaPtr = thrust::raw_pointer_cast(listSigmaPtr.data());

			int* dualNumbersCUDAPtr = thrust::raw_pointer_cast(dualNumbersCUDA.data());

			dualLabelingProxCUDA << <(int)ceil(numElements / 512), 512 >> >(YPtr,YTildePtr,FPtr,SigmaPtr,dualNumbersCUDAPtr,numElements,numDualVars);
		#else
			int numElements = (int)data->yTilde[dualNumbers[0]].size();
			
			int numDualVars = (int)dualNumbers.size();

			//create vector of pointers:
			#pragma omp parallel
			{
				std::vector<T> tmpVector(numDualVars);
				std::vector<T> tmpVector2(numDualVars);

				T sumY, tOpt;

				//do this for every element:
				#pragma omp for
				for (int i = 0; i < numElements; i++)
				{
					//copy from yTilde
					for (int j = 0; j < numDualVars; j++)
					{
						tmpVector[j] = (data->yTilde[dualNumbers[j]][i] - fList[j][i]) / data->sigmaElt[dualNumbers[j]][i];
						tmpVector2[j] = tmpVector[j];
					}

					//sort y values 
					std::sort(tmpVector.begin(), tmpVector.end());

					sumY = (T)0;

					int j = numDualVars - 2;
					while (j >= 0)
					{
						sumY = sumY + tmpVector[j + 1];

						tOpt = (sumY - 1) / (numDualVars - (j + 1));

						if (tOpt >= tmpVector[j])
						{
							break;
						}
						else
						{
							j = j - 1;
						}
					}

					if (j < 0)
					{
						tOpt = (sumY + tmpVector[0] - 1) / numDualVars;
					}

					//write result
					for (int j = 0; j < numDualVars; j++)
					{
						data->y[dualNumbers[j]][i] = data->yTilde[dualNumbers[j]][i] - data->sigmaElt[dualNumbers[j]][i] * std::max(tmpVector2[j] - tOpt, (T)0);
					}
				}
			}
		#endif
	}
};

#endif
