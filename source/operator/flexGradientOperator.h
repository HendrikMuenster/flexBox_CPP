#ifndef flexGradientOperator_H
#define flexGradientOperator_H

#include <vector>
#include "flexLinearOperator.h"

template<typename T>
class flexGradientOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	std::vector<int> inputDimension;

	int* inputDimensionPtr;
	int gradDirection;
	int type;
	bool transposed;
	int numberDimensions;

public:

	//type:
	// 0 = forward
	// 1 = backward
	flexGradientOperator(std::vector<int> AInputDimension, int aGradDirection, int aType, bool _minus) : flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), gradientOp, _minus)
	{

		this->gradDirection = aGradDirection;
		this->type = aType;
		this->numberDimensions = static_cast<int>(AInputDimension.size());

		this->inputDimension = AInputDimension;
		this->inputDimensionPtr = this->inputDimension.data();

	};

	flexGradientOperator<T>* copy()
	{

		std::vector<int> dimsCopy;
		dimsCopy.resize(this->inputDimension.size());

		std::copy(this->inputDimension.begin(), this->inputDimension.end(), dimsCopy.begin());

		return new flexGradientOperator<T>(dimsCopy, this->gradDirection, this->type, this->isMinus);
	}

	//apply linear operator to vector
	void times(bool transposed, const Tdata &input, Tdata &output)
	{

	}



	void dxp2d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeY = this->inputDimension[1];
		int sizeX = this->inputDimension[0] - 1;

		#pragma omp parallel for
		for (int j = 0; j < sizeY; ++j)
		{
			for (int i = 0; i < sizeX; ++i)
			{
				const int tmpIndex = this->index2DtoLinear(i, j);

				switch (s)
				{
					case PLUS:
					{
						output[tmpIndex] += input[tmpIndex + 1] - input[tmpIndex];
						break;
					}
					case MINUS:
					{
						output[tmpIndex] -= input[tmpIndex + 1] - input[tmpIndex];
						break;
					}
					case EQUALS:
					{
						output[tmpIndex] = input[tmpIndex + 1] - input[tmpIndex];
						break;
					}
				}
			}
		}
	}

	void dyp2d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeY = this->inputDimension[1] - 1;
		int sizeX = this->inputDimension[0];

		#pragma omp parallel for
		for (int j = 0; j < sizeY; ++j)
		{
			for (int i = 0; i < sizeX; ++i)
			{
				const int tmpIndex = this->index2DtoLinear(i, j);

				switch (s)
				{
					case PLUS:
					{
						output[tmpIndex] += input[tmpIndex + sizeX] - input[tmpIndex];
						break;
					}
					case MINUS:
					{
						output[tmpIndex] -= input[tmpIndex + sizeX] - input[tmpIndex];
						break;
					}
					case EQUALS:
					{
						output[tmpIndex] = input[tmpIndex + sizeX] - input[tmpIndex];
						break;
					}
				}
			}
		}
	}

	void dxp2dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeY = this->inputDimension[1];
		int sizeX = this->inputDimension[0] - 1;

		#pragma omp parallel for
		for (int j = 0; j < sizeY; ++j)
		{
			for (int i = 1; i < sizeX; ++i)
			{
				int tmpIndex = this->index2DtoLinear(i, j);

				switch (s)
				{
					case PLUS:
					{
						output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - 1]);
						break;
					}
					case MINUS:
					{
						output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - 1]);
						break;
					}
					case EQUALS:
					{
						output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - 1]);
						break;
					}
				}
			}
		}

		for (int j = 0; j < this->inputDimension[1]; ++j)
		{
			switch (s)
			{
				case PLUS:
				{
					output[this->index2DtoLinear(0, j)] += -input[this->index2DtoLinear(0, j)];
					output[this->index2DtoLinear(this->inputDimension[0] - 1, j)] += input[this->index2DtoLinear(this->inputDimension[0] - 2, j)];
					break;
				}
				case MINUS:
				{
					output[this->index2DtoLinear(0, j)] -= -input[this->index2DtoLinear(0, j)];
					output[this->index2DtoLinear(this->inputDimension[0] - 1, j)] -= input[this->index2DtoLinear(this->inputDimension[0] - 2, j)];
					break;
				}
				case EQUALS:
				{
					output[this->index2DtoLinear(0, j)] = -input[this->index2DtoLinear(0, j)];
					output[this->index2DtoLinear(this->inputDimension[0] - 1, j)] = input[this->index2DtoLinear(this->inputDimension[0] - 2, j)];
					break;
				}
			}
		}
	}

	void dyp2dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeY = this->inputDimension[1] - 1;
		int sizeX = this->inputDimension[0];

		#pragma omp parallel for
		for (int j = 1; j < sizeY; ++j)
		{
			for (int i = 0; i < sizeX; ++i)
			{
				int tmpIndex = this->index2DtoLinear(i, j);

				switch (s)
				{
					case PLUS:
					{
						output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - sizeX]);
						break;
					}
					case MINUS:
					{
						output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - sizeX]);
						break;
					}
					case EQUALS:
					{
						output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - sizeX]);
						break;
					}
				}
			}
		}

		for (int i = 0; i < this->inputDimension[0]; ++i)
		{
			switch (s)
			{
			case PLUS:
			{
				output[this->index2DtoLinear(i, 0)] += -input[this->index2DtoLinear(i, 0)];
				output[this->index2DtoLinear(i, this->inputDimension[1] - 1)] += input[this->index2DtoLinear(i, this->inputDimension[1] - 2)];
				break;
			}
			case MINUS:
			{
				output[this->index2DtoLinear(i, 0)] -= -input[this->index2DtoLinear(i, 0)];
				output[this->index2DtoLinear(i, this->inputDimension[1] - 1)] -= input[this->index2DtoLinear(i, this->inputDimension[1] - 2)];
				break;
			}
			case EQUALS:
			{
				output[this->index2DtoLinear(i, 0)] = -input[this->index2DtoLinear(i, 0)];
				output[this->index2DtoLinear(i, this->inputDimension[1] - 1)] = input[this->index2DtoLinear(i, this->inputDimension[1] - 2)];
				break;
			}
			}
		}
	}


	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
        #ifdef __CUDACC__
			dim3 block2d = dim3(32, 16, 1);
			dim3 grid2d = dim3((this->inputDimension[0] + block2d.x - 1) / block2d.x, (this->inputDimension[1] + block2d.y - 1) / block2d.y, 1);

			T* ptrOutput = thrust::raw_pointer_cast(output.data());
			const T* ptrInput = thrust::raw_pointer_cast(input.data());
		#endif
		
        mySign s;
        int s2;
		
		if (this->isMinus)
        {
            s = MINUS;
            s2 = SIGN_MINUS;
		}
        else
        {
            s = PLUS;
            s2 = SIGN_PLUS;
        }

		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					#ifdef __CUDACC__
						dxp2dCUDA << <grid2d, block2d >> >(ptrOutput,ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dxp2d(input,output,s);
					#endif
				}
				else
				{
					#ifdef __CUDACC__
						dxp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dxp2dTransposed(input, output, s);
					#endif
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					#ifdef __CUDACC__
						dyp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dyp2d(input, output, s);
					#endif
				}
				else
				{
					#ifdef __CUDACC__
						dyp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dyp2dTransposed(input, output, s);
					#endif
				}
			}
		}
		else
		{
            printf("Gradient not implemented for dim!=2");
			//todo
		}
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		#ifdef __CUDACC__
			dim3 block2d = dim3(32, 16, 1);
			dim3 grid2d = dim3((this->inputDimension[0] + block2d.x - 1) / block2d.x, (this->inputDimension[1] + block2d.y - 1) / block2d.y, 1);

			T* ptrOutput = thrust::raw_pointer_cast(output.data());
			const T* ptrInput = thrust::raw_pointer_cast(input.data());
		#endif
		
        mySign s;
        int s2;
		
		if (this->isMinus)
        {
            s = MINUS;
            s2 = SIGN_MINUS;
		}
        else
        {
            s = PLUS;
            s2 = SIGN_PLUS;
        }

		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					#ifdef __CUDACC__
						dxp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dxp2d(input, output, s);
					#endif
				}
				else
				{
					#ifdef __CUDACC__
						dxp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dxp2dTransposed(input, output, s);
					#endif
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					#ifdef __CUDACC__
						dyp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dyp2d(input, output, s);
					#endif
				}
				else
				{
					#ifdef __CUDACC__
						dyp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
					#else
						this->dyp2dTransposed(input, output, s);
					#endif
				}
			}
		}
		else
        {
            printf("Gradient not implemented for dim!=2");
			//todo
		}
	}

	T getMaxRowSumAbs(bool transposed)
	{
		//row sum of absolute values is at maximum 2
		return static_cast<T>(2);
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
		std::vector<T> result(this->getNumRows(),(T)2);

		return result;
	}

	int index2DtoLinear(int i, int j)
	{
		return (i + j*this->inputDimension[0]);
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		thrust::device_vector<T> result(this->getNumRows(), (T)2);

		return result;
	}
	#endif
};

#endif
