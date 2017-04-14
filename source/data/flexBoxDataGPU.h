#ifndef flexBoxDataGPU_H
#define flexBoxDataGPU_H

#include "flexBoxData.h"

//! FlexBox data class if using the CUDA
/*!
	flexBoxDataCPU is an internal class for storing and maintaining data
	for the optimization algorithm if using the CUDA version.
	This class should not be used directly.
*/
template <class T>
class flexBoxDataGPU : public flexBoxData<T>
{
public:
	flexBoxDataGPU() : flexBoxData<T>(){}

	std::vector<T> getPrimal(int i)
	{
		std::vector<T> xTmp(this->x[i].size());
		thrust::copy(this->x[i].begin(), this->x[i].end(), xTmp.begin());
		return xTmp;
	}

	std::vector<T> getDual(int i)
	{
		std::vector<T> yTmp(this->y[i].size());
		thrust::copy(this->y[i].begin(), this->y[i].end(), yTmp.begin());
		return yTmp;
	}

	void setPrimal(int i, std::vector<T> input)
	{
		thrust::copy(input.begin(), input.end(), this->x[i].begin());
		thrust::copy(input.begin(), input.end(), this->xOld[i].begin());
		thrust::copy(input.begin(), input.end(), this->xBar[i].begin());
	}

	void setDual(int i, std::vector<T> input)
	{
		thrust::copy(input.begin(), input.end(), this->y[i].begin());
		thrust::copy(input.begin(), input.end(), this->yOld[i].begin());
		thrust::copy(input.begin(), input.end(), this->yTilde[i].begin());
	}
};

#endif //flexBoxDataGPU_H
