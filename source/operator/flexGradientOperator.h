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
	int gradDirection;
	gradientType type;
	int numberDimensions;

public:

	//type: forward, backward, central
	flexGradientOperator(std::vector<int> AInputDimension, int aGradDirection, gradientType aType, bool _minus) : 
		inputDimension(AInputDimension), 
		gradDirection(aGradDirection),
		type(aType),
		numberDimensions(static_cast<int>(AInputDimension.size())), flexLinearOperator<T>(vectorProduct(AInputDimension), vectorProduct(AInputDimension), gradientOp, _minus)
	{};

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

	//
	//
	//  3d cases
	//
	//

	void dxp3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeZ = this->inputDimension[2];
		int sizeY = this->inputDimension[1];
		int sizeX = this->inputDimension[0] - 1;

		#pragma omp parallel for
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					const int tmpIndex = this->index3DtoLinear(i, j, k);

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
	}

	void dyp3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeZ = this->inputDimension[2]; 
		int sizeY = this->inputDimension[1] - 1;
		int sizeX = this->inputDimension[0];

		#pragma omp parallel for
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					const int tmpIndex1 = this->index3DtoLinear(i, j, k);
					const int tmpIndex2 = this->index3DtoLinear(i, j + 1, k);

					switch (s)
					{
					case PLUS:
					{
						output[tmpIndex1] += input[tmpIndex2] - input[tmpIndex1];
						break;
					}
					case MINUS:
					{
						output[tmpIndex1] -= input[tmpIndex2] - input[tmpIndex1];
						break;
					}
					case EQUALS:
					{
						output[tmpIndex1] = input[tmpIndex2] - input[tmpIndex1];
						break;
					}
					}
				}
			}
		}
	}

	void dzp3d(const Tdata &input, Tdata &output, mySign s)
	{
		int sizeZ = this->inputDimension[2] - 1;
		int sizeY = this->inputDimension[1];
		int sizeX = this->inputDimension[0];

		#pragma omp parallel for
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					const int tmpIndex1 = this->index3DtoLinear(i, j, k);
					const int tmpIndex2 = this->index3DtoLinear(i, j, k + 1);

					switch (s)
					{
					case PLUS:
					{
						output[tmpIndex1] += input[tmpIndex2] - input[tmpIndex1];
						break;
					}
					case MINUS:
					{
						output[tmpIndex1] -= input[tmpIndex2] - input[tmpIndex1];
						break;
					}
					case EQUALS:
					{
						output[tmpIndex1] = input[tmpIndex2] - input[tmpIndex1];
						break;
					}
					}
				}
			}
		}
	}

	void dxp3dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		const int sizeZ = this->inputDimension[2];
		const int sizeY = this->inputDimension[1];
		const int sizeX = this->inputDimension[0] - 1;

		#pragma omp parallel for
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 1; i < sizeX; ++i)
				{
					int tmpIndex = this->index3DtoLinear(i, j, k);

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
		}

		#pragma omp parallel for
		for (int k = 0; k < this->inputDimension[2]; ++k)
		{
			for (int j = 0; j < this->inputDimension[1]; ++j)
			{
				const int index1 = this->index3DtoLinear(0, j, k);
				const int index2 = this->index3DtoLinear(this->inputDimension[0] - 1, j, k);
				const int index3 = this->index3DtoLinear(this->inputDimension[0] - 2, j, k);

				switch (s)
				{
				case PLUS:
				{
					output[index1] += -input[index1];
					output[index2] += input[index3];
					break;
				}
				case MINUS:
				{
					output[index1] -= -input[index1];
					output[index2] -= input[index3];
					break;
				}
				case EQUALS:
				{
					output[index1] = -input[index1];
					output[index2] = input[index3];
					break;
				}
				}
			}
		}
	}

	void dyp3dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		const int sizeZ = this->inputDimension[2];
		const int sizeY = this->inputDimension[1] - 1;
		const int sizeX = this->inputDimension[0];

		#pragma omp parallel for
		for (int k = 0; k < sizeZ; ++k)
		{
			for (int j = 1; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					const int tmpIndex1 = this->index3DtoLinear(i, j, k);
					const int tmpIndex2 = this->index3DtoLinear(i, j - 1, k);

					switch (s)
					{
					case PLUS:
					{
						output[tmpIndex1] += -(input[tmpIndex1] - input[tmpIndex2]);
						break;
					}
					case MINUS:
					{
						output[tmpIndex1] -= -(input[tmpIndex1] - input[tmpIndex2]);
						break;
					}
					case EQUALS:
					{
						output[tmpIndex1] = -(input[tmpIndex1] - input[tmpIndex2]);
						break;
					}
					}
				}
			}
		}

		#pragma omp parallel for
		for (int k = 0; k < this->inputDimension[2]; ++k)
		{
			for (int i = 0; i < this->inputDimension[0]; ++i)
			{
				const int index1 = this->index3DtoLinear(i, 0, k);
				const int index2 = this->index3DtoLinear(i, this->inputDimension[1] - 1, k);
				const int index3 = this->index3DtoLinear(i, this->inputDimension[1] - 2, k);

				switch (s)
				{
				case PLUS:
				{
					output[index1] += -input[index1];
					output[index2] += input[index3];
					break;
				}
				case MINUS:
				{
					output[index1] -= -input[index1];
					output[index2] -= input[index3];
					break;
				}
				case EQUALS:
				{
					output[index1] = -input[index1];
					output[index2] = input[index3];
					break;
				}
				}
			}
		}
	}

	void dzp3dTransposed(const Tdata &input, Tdata &output, mySign s)
	{
		const int sizeZ = this->inputDimension[2] - 1;
		const int sizeY = this->inputDimension[1];
		const int sizeX = this->inputDimension[0];

		#pragma omp parallel for
		for (int k = 1; k < sizeZ; ++k)
		{
			for (int j = 0; j < sizeY; ++j)
			{
				for (int i = 0; i < sizeX; ++i)
				{
					const int tmpIndex1 = this->index3DtoLinear(i, j, k);
					const int tmpIndex2 = this->index3DtoLinear(i, j, k - 1);

					switch (s)
					{
					case PLUS:
					{
						output[tmpIndex1] += -(input[tmpIndex1] - input[tmpIndex2]);
						break;
					}
					case MINUS:
					{
						output[tmpIndex1] -= -(input[tmpIndex1] - input[tmpIndex2]);
						break;
					}
					case EQUALS:
					{
						output[tmpIndex1] = -(input[tmpIndex1] - input[tmpIndex2]);
						break;
					}
					}
				}
			}
		}

		#pragma omp parallel for
		for (int j = 0; j < this->inputDimension[1]; ++j)
		{
			for (int i = 0; i < this->inputDimension[0]; ++i)
			{
				const int index1 = this->index3DtoLinear(i, j, 0);
				const int index2 = this->index3DtoLinear(i, j, this->inputDimension[2] - 1);
				const int index3 = this->index3DtoLinear(i, j, this->inputDimension[2] - 2);

				switch (s)
				{
				case PLUS:
				{
					output[index1] += -input[index1];
					output[index2] += input[index3];
					break;
				}
				case MINUS:
				{
					output[index1] -= -input[index1];
					output[index2] -= input[index3];
					break;
				}
				case EQUALS:
				{
					output[index1] = -input[index1];
					output[index2] = input[index3];
					break;
				}
				}
			}
		}
	}

	//2d cases
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
		
		// flip sign and transposed
		if (this->type == backward)
		{
			transposed = !transposed;
			if (s == MINUS)
			{
				s = PLUS;
				s2 = SIGN_PLUS;
			}
			else
			{
				s = MINUS;
				s2 = SIGN_MINUS;
			}
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
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
#ifdef __CUDACC__
//					dxp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
#else
					this->dxp3d(input, output, s);
#endif
				}
				else
				{
#ifdef __CUDACC__
//					dxp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
#else
					this->dxp3dTransposed(input, output, s);
#endif
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
#ifdef __CUDACC__
//					dyp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
#else
					this->dyp3d(input, output, s);
#endif
				}
				else
				{
#ifdef __CUDACC__
//					dyp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
#else
					this->dyp3dTransposed(input, output, s);
#endif
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
#ifdef __CUDACC__
					//					dyp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
#else
					this->dzp3d(input, output, s);
#endif
				}
				else
				{
#ifdef __CUDACC__
					//					dyp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
#else
					this->dzp3dTransposed(input, output, s);
#endif
				}
			}
		}
		else
		{
			printf("Gradient not implemented for dim!={2,3}\n");
			//todo
		}
	}

	void doTimesMinusCPU(bool transposed, const Tdata &input, Tdata &output)
	{
		mySign s;

		if (this->isMinus)
		{
			s = MINUS;
		}
		else
		{
			s = PLUS;
		}

		// flip sign and transposed
		if (this->type == backward)
		{
			transposed = !transposed;
			if (s == MINUS)
			{
				s = PLUS;
			}
			else
			{
				s = MINUS;
			}
		}

		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					this->dxp2d(input, output, s);
				}
				else
				{
					this->dxp2dTransposed(input, output, s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					this->dyp2d(input, output, s);
				}
				else
				{
					this->dyp2dTransposed(input, output, s);
				}
			}
		}
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					this->dxp3d(input, output, s);
				}
				else
				{
					this->dxp3dTransposed(input, output, s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					this->dyp3d(input, output, s);
				}
				else
				{
					this->dyp3dTransposed(input, output, s);
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
					this->dzp3d(input, output, s);
				}
				else
				{
					this->dzp3dTransposed(input, output, s);
				}
			}
		}
		else
		{
			printf("Gradient not implemented for dim!={2,3}\n");
			//todo
		}
	}

	#ifdef __CUDACC__
	void doTimesMinusCUDA(bool transposed, const Tdata &input, Tdata &output)
	{

		dim3 block2d = dim3(32, 16, 1);
		dim3 grid2d = dim3((this->inputDimension[0] + block2d.x - 1) / block2d.x, (this->inputDimension[1] + block2d.y - 1) / block2d.y, 1);

		T* ptrOutput = thrust::raw_pointer_cast(output.data());
		const T* ptrInput = thrust::raw_pointer_cast(input.data());

		int s2;

		if (this->isMinus)
		{
			s2 = SIGN_MINUS;
		}
		else
		{
			s2 = SIGN_PLUS;
		}

		// flip sign and transposed
		if (this->type == backward)
		{
			transposed = !transposed;
			if (s == MINUS)
			{
				s2 = SIGN_PLUS;
			}
			else
			{
				s2 = SIGN_MINUS;
			}
		}

		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					dxp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
				else
				{
					dxp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					dyp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
				else
				{
					dyp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
			}
		}
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					//					dxp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
				else
				{
					//					dxp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					//					dyp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
				else
				{
					//					dyp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
					//					dyp2dCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
				else
				{
					//					dyp2dTransposedCUDA << <grid2d, block2d >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s2);
				}
			}
		}
		else
		{
			printf("Gradient not implemented for dim!={2,3}\n");
			//todo
		}
	}
	#endif

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		#ifdef __CUDACC__
		this->doTimesMinusCUDA(transposed,input, output);
		#else
		this->doTimesMinusCPU(transposed, input, output);
		#endif
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

	int index3DtoLinear(int i, int j, int k)
	{
		return (i + j*this->inputDimension[0] + k*this->inputDimension[0] * this->inputDimension[1]);
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
