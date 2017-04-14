#ifndef flexSuperpixelOperator_H
#define flexSuperpixelOperator_H

#include <vector>
#include "flexLinearOperator.h"

//! represents a superpixel operator
/*!
	downsamples data of size upsamplingFactor * targetDimension size to targetDimension
*/
template<typename T>
class flexSuperpixelOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<int> targetDimension;
	T upsamplingFactor;
public:

	//! initializes the superpixel operator
	/*!
		\param aTargetDimension TODO
		\param aUpsamplingFactor TODO
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexSuperpixelOperator(std::vector<int> aTargetDimension, T aUpsamplingFactor, bool aMinus) : flexLinearOperator<T>((int)(vectorProduct(aTargetDimension)), (int)(vectorProduct(aTargetDimension)*aUpsamplingFactor*aUpsamplingFactor), superpixelOp, aMinus)
	{
		this->targetDimension.resize(aTargetDimension.size());

        this->targetDimension = aTargetDimension;
		this->upsamplingFactor = aUpsamplingFactor;
	};

	flexSuperpixelOperator<T>* copy()
	{
		return new flexSuperpixelOperator<T>(this->targetDimension, this->upsamplingFactor, this->isMinus);
	}



	//to implement
	void times(bool transposed, const Tdata &input, Tdata &output)
	{

	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
        if (this->isMinus)
        {
            doTimes(transposed,input,output, MINUS);
        }
        else
        {
            doTimes(transposed,input,output, PLUS);
        }
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
        if (this->isMinus)
        {
            doTimes(transposed,input,output, PLUS);
        }
        else
        {
            doTimes(transposed,input,output, MINUS);
        }
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		if (transposed)
		{
			return std::vector<T>(this->getNumCols(), (T)1 / (T)(this->upsamplingFactor*this->upsamplingFactor));
		}
		else
		{
			return std::vector<T>(this->getNumRows(), (T)1);
		}
	}

	T getMaxRowSumAbs(bool transposed)
	{
		if (transposed)
		{
			return (T)1 / (T)(this->upsamplingFactor*this->upsamplingFactor);
		}
		else
		{
			return (T)1;
		}
	}

    #ifdef __CUDACC__
    thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		Tdata result(this->getNumRows(),(T)1);

		return result;
	}
    #endif

	private:
		int indexI(int index, int sizeX)
		{
			return index % sizeX;
		}

		int indexJ(int index, int sizeX, int sizeY)
		{
			return (index / sizeX) % sizeY;
		}

		int index2DtoLinear(int i, int j, int sizeY)
		{
			return (i*sizeY + j);
		}

		void calcTimes(const Tdata &input, Tdata &output, mySign signRule)
		{

			T factor = (T)1 / (this->upsamplingFactor*this->upsamplingFactor);

			int iOuterEnd = targetDimension[0];
			int jOuterEnd = targetDimension[1];

			int sizeY = targetDimension[1] * (int)this->upsamplingFactor;

			#pragma omp parallel for
			for (int i = 0; i < iOuterEnd; ++i)
			{
				for (int j = 0; j < jOuterEnd; ++j)
				{
					//printf("Output: (%d,%d) : %d\n", i, j, index2DtoLinear(i, j, this->targetDimension[1]));

					int outputIndex = index2DtoLinear(i, j, targetDimension[1]);

					int iInnerStart = i*(int)this->upsamplingFactor;
					int iInnerEnd = (i + 1)*(int)this->upsamplingFactor;

					int jInnerStart = j*(int)this->upsamplingFactor;
					int jInnerEnd = (j + 1)*(int)this->upsamplingFactor;

					T tmpResult = (T)0;

					for (int iInner = iInnerStart; iInner < iInnerEnd; ++iInner)
					{
						for (int jInner = jInnerStart; jInner < jInnerEnd; ++jInner)
						{
							int inputIndex = index2DtoLinear(iInner, jInner, sizeY);

							tmpResult += input[inputIndex];
							/*printf("Inner: (%d,%d) : %d\n", iInner, jInner, inputIndex);

							int innerJ = indexI(inputIndex, this->targetDimension[0] * this->upsamplingFactor);
							int innerI = indexJ(inputIndex, this->targetDimension[0] * this->upsamplingFactor, this->targetDimension[1] * this->upsamplingFactor);

							printf("Back: (%d,%d) \n", innerI, innerJ);

							int backI = innerI / this->upsamplingFactor;
							int backJ = innerJ / this->upsamplingFactor;

							printf("BackInner: (%d,%d) \n", backI, backJ);

							if (backI != i || backJ != j)
							{
								mexErrMsgTxt("PROBLEM!!!\n");
							}*/
						}
					}

					switch (signRule)
					{
						case PLUS:
						{
							output[outputIndex] += factor*tmpResult;
							break;
						}
						case MINUS:
						{
							output[outputIndex] -= factor*tmpResult;
							break;
						}
						case EQUALS:
						{
							output[outputIndex] = factor*tmpResult;
							break;
						}
					}
				}
			}
		}

		void calcTimesTransposed(const Tdata &input, Tdata &output, mySign signRule)
		{
			T factor = (T)1 / (this->upsamplingFactor*this->upsamplingFactor);

			int sizeX = targetDimension[0] * (int)this->upsamplingFactor;
			int sizeY = targetDimension[1] * (int)this->upsamplingFactor;

			#pragma omp parallel for
			for (int i = 0; i < sizeX; ++i)
			{
				for (int j = 0; j < sizeY; ++j)
				{
					int inputIndex = index2DtoLinear(i, j, sizeY);

					//int innerJ = indexI(inputIndex, this->targetDimension[0] * this->upsamplingFactor);
					//int innerI = indexJ(inputIndex, this->targetDimension[0] * this->upsamplingFactor, this->targetDimension[1] * this->upsamplingFactor);

					//printf("Back: (%d,%d) \n", innerI, innerJ);

					int backI = i / (int)this->upsamplingFactor;
					int backJ = j / (int)this->upsamplingFactor;


					int outputIndex = index2DtoLinear(backI, backJ, targetDimension[1]);

					//printf("Back: (%d,%d) %d,%d \n", backI, backJ, inputIndex, outputIndex);

					switch (signRule)
					{
						case PLUS:
						{
							output[inputIndex] += factor*input[outputIndex];
							break;
						}
						case MINUS:
						{
							output[inputIndex] -= factor*input[outputIndex];
							break;
						}
						case EQUALS:
						{
							output[inputIndex] = factor*input[outputIndex];
							break;
						}
					}
				}
			}
		}

		void doTimes(bool transposed, const Tdata &input, Tdata &output, mySign signRule)
		{
			if (transposed)
			{
				calcTimesTransposed(input, output, signRule);
			}
			else
			{
				calcTimes(input, output, signRule);
			}
	  }
};

#endif
