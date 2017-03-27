#ifndef flexGradientOperator_H
#define flexGradientOperator_H

#include <vector>
#include "flexLinearOperator.h"

#ifdef __CUDACC__
//
// 2d kernels
//
template<typename T>
__global__ void dxp2dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= sizeX || y >= sizeY)
		return;

	const size_t tmpIndex = x + y * sizeX;

	if (x < sizeX - 1)
	{
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

template<typename T>
__global__ void dxp2dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= sizeX || y >= sizeY)
		return;

	const size_t tmpIndex = x + y * sizeX;

	if (x < sizeX - 1 && x > 0)
	{
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
	else if (x == 0)
	{
		switch (s)
		{
		case PLUS:
		{
			output[tmpIndex] += -(input[tmpIndex]);
			break;
		}
		case MINUS:
		{
			output[tmpIndex] -= -(input[tmpIndex]);
			break;
		}
		case EQUALS:
		{
			output[tmpIndex] = -(input[tmpIndex]);
			break;
		}
		}
	}
	else
	{
		switch (s)
		{
		case PLUS:
		{
			output[tmpIndex] += (input[tmpIndex - 1]);
			break;
		}
		case MINUS:
		{
			output[tmpIndex] -= (input[tmpIndex - 1]);
			break;
		}
		case EQUALS:
		{
			output[tmpIndex] = (input[tmpIndex - 1]);
			break;
		}
		}
	}
}

template<typename T>
__global__ void dyp2dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= sizeX || y >= sizeY)
		return;

	const size_t tmpIndex = x + y * sizeX;

	if (y < sizeY - 1)
	{
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

template<typename T>
__global__ void dyp2dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, mySign s)
{
	const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= sizeX || y >= sizeY)
		return;

	const size_t tmpIndex = x + y * sizeX;

	if (y < sizeY - 1 && y > 0)
	{
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
	else if (y == 0)
	{
		switch (s)
		{
		case PLUS:
		{
			output[tmpIndex] += -(input[tmpIndex]);
			break;
		}
		case MINUS:
		{
			output[tmpIndex] -= -(input[tmpIndex]);
			break;
		}
		case EQUALS:
		{
			output[tmpIndex] = -(input[tmpIndex]);
			break;
		}
		}
	}
	else
	{
		switch (s)
		{
		case PLUS:
		{
			output[tmpIndex] += (input[tmpIndex - sizeX]);
			break;
		}
		case MINUS:
		{
			output[tmpIndex] -= (input[tmpIndex - sizeX]);
			break;
		}
		case EQUALS:
		{
			output[tmpIndex] = (input[tmpIndex - sizeX]);
			break;
		}
		}
	}
}

//
// 3d kernels
// 

__device__
int getGlobalIdx_3D_3D(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

template<typename T>
__global__ void dxp3dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

	const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ)
		return;

	if (x < sizeX - 1)
	{
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

template<typename T>
__global__ void dxp3dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

	const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ)
		return;

	if (x < sizeX - 1 && x > 0)
	{
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
	else if (x == 0)
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += -(input[tmpIndex]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= -(input[tmpIndex]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = -(input[tmpIndex]);
				break;
			}
		}
	}
	else
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += (input[tmpIndex - 1]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= (input[tmpIndex - 1]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = (input[tmpIndex - 1]);
				break;
			}
		}
	}
}

template<typename T>
__global__ void dyp3dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

	const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ)
		return;
	
	if (y < sizeY - 1)
	{
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

template<typename T>
__global__ void dyp3dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

	const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ)
		return;
	
	if (y < sizeY - 1 && y > 0)
	{
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
	else if (y == 0)
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += -(input[tmpIndex]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= -(input[tmpIndex]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = -(input[tmpIndex]);
				break;
			}
		}
	}
	else
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += (input[tmpIndex - sizeX]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= (input[tmpIndex - sizeX]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = (input[tmpIndex - sizeX]);
				break;
			}
		}
	}
}

template<typename T>
__global__ void dzp3dCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

	const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ)
		return;
	
	if (z < sizeZ - 1)
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += input[tmpIndex + sizeX * sizeY] - input[tmpIndex];
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= input[tmpIndex + sizeX * sizeY] - input[tmpIndex];
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = input[tmpIndex + sizeX * sizeY] - input[tmpIndex];
				break;
			}
		}
	}
}

template<typename T>
__global__ void dzp3dTransposedCUDA(T* output, const T* input, const size_t sizeX, const size_t sizeY, const size_t sizeZ, mySign s)
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

	const size_t tmpIndex = x + sizeX * y + sizeX * sizeY * z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ)
		return;
	
	if (z < sizeZ - 1 && z > 0)
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += -(input[tmpIndex] - input[tmpIndex - sizeX * sizeY]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= -(input[tmpIndex] - input[tmpIndex - sizeX * sizeY]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = -(input[tmpIndex] - input[tmpIndex - sizeX * sizeY]);
				break;
			}
		}
	}
	else if (z == 0)
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += -(input[tmpIndex]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= -(input[tmpIndex]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = -(input[tmpIndex]);
				break;
			}
		}
	}
	else
	{
		switch (s)
		{
			case PLUS:
			{
				output[tmpIndex] += (input[tmpIndex - sizeX * sizeY]);
				break;
			}
			case MINUS:
			{
				output[tmpIndex] -= (input[tmpIndex - sizeX * sizeY]);
				break;
			}
			case EQUALS:
			{
				output[tmpIndex] = (input[tmpIndex - sizeX * sizeY]);
				break;
			}
		}
	}
}

#endif

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

	//
	//
	//  3d cases
	//
	//

	void updateValue(T* ptr, mySign s, T value)
	{
		switch (s)
		{
		case PLUS:
		{
			*ptr += value;
			break;
		}
		case MINUS:
		{
			*ptr -= value;
			break;
		}
		case EQUALS:
		{
			*ptr = value;
			break;
		}
		}
	}

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

					this->updateValue(&output[tmpIndex], s, input[tmpIndex + 1] - input[tmpIndex]);
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

					this->updateValue(&output[tmpIndex1], s, input[tmpIndex2] - input[tmpIndex1]);
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

					this->updateValue(&output[tmpIndex1], s, input[tmpIndex2] - input[tmpIndex1]);
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
					
					this->updateValue(&output[tmpIndex], s, -(input[tmpIndex] - input[tmpIndex - 1]));
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

				this->updateValue(&output[index1], s, -input[index1]);
				this->updateValue(&output[index2], s, input[index3]);
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

					this->updateValue(&output[tmpIndex1], s, -(input[tmpIndex1] - input[tmpIndex2]));
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

				this->updateValue(&output[index1], s, -input[index1]);
				this->updateValue(&output[index2], s, input[index3]);
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

					this->updateValue(&output[tmpIndex1], s, -(input[tmpIndex1] - input[tmpIndex2]));
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

				this->updateValue(&output[index1], s, -input[index1]);
				this->updateValue(&output[index2], s, input[index3]);
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

	void doTimesCPU(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
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
	void doTimesCUDA(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		size_t sizeX = this->inputDimension[0];
		size_t sizeY = 1;
		size_t sizeZ = 1;

		if (this->inputDimension.size() > 1)
		{
			sizeY = this->inputDimension[1];
		}
		if (this->inputDimension.size() > 2)
		{
			sizeZ = this->inputDimension[2];
		}

		dim3 block = dim3(32,16,1);
		dim3 grid = dim3((sizeX + block.x - 1) / block.x, (sizeY + block.y - 1) / block.y, (sizeZ + block.z - 1) / block.z);

		T* ptrOutput = thrust::raw_pointer_cast(output.data());
		const T* ptrInput = thrust::raw_pointer_cast(input.data());

		if (this->inputDimension.size() == 2)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					dxp2dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
				else
				{
					dxp2dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					dyp2dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
				else
				{
					dyp2dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], s);
				}
			}
		}
		else if (this->inputDimension.size() == 3)
		{
			if (this->gradDirection == 0)
			{
				if (transposed == false)
				{
					dxp3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dxp3dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
			else if (this->gradDirection == 1)
			{
				if (transposed == false)
				{
					dyp3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dyp3dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
			}
			else if (this->gradDirection == 2)
			{
				if (transposed == false)
				{
					dzp3dCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
				}
				else
				{
					dzp3dTransposedCUDA << <grid, block >> >(ptrOutput, ptrInput, this->inputDimension[0], this->inputDimension[1], this->inputDimension[2], s);
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

	void doTimes(bool transposed, const Tdata &input, Tdata &output, mySign s)
	{
		if (this->isMinus && s == PLUS)
		{
			s = MINUS;
		}
		else if (this->isMinus && s == MINUS)
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

		#ifdef __CUDACC__
			this->doTimesCUDA(transposed, input, output, s);
		#else
			this->doTimesCPU(transposed, input, output, s);
		#endif
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, PLUS);
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, MINUS);
	}

	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(transposed, input, output, EQUALS);
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
