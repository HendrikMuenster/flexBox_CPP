#ifndef flexProxDualLabeling_H
#define flexProxDualLabeling_H

#include "flexProx.h"

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
	
	//put cuda functor here
    #ifdef __CUDACC__
    #endif
	
	

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
	}

	void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList)
	{
#ifdef __CUDACC__
			printf("flexProxDualLabeling not implemented for CUDA\n");
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
