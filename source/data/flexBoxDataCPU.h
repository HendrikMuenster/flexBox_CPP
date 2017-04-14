#ifndef flexBoxDataCPU_H
#define flexBoxDataCPU_H

#include "flexBoxData.h"

//! FlexBox data class if using the non-CUDA version
/*!
	flexBoxDataCPU is an internal class for storing and maintaining data
	for the optimization algorithm if using the non-CUDA version.
	This class should not be used directly.
*/
template <class T>
class flexBoxDataCPU : public flexBoxData<T>
{
public:
	flexBoxDataCPU() : flexBoxData<T>(){}

	std::vector<T> getPrimal(int i)
	{
		return this->x[i];
	}

	std::vector<T> getDual(int i)
	{
		return this->y[i];
	}

	void setPrimal(int i, std::vector<T> input)
	{
		this->x[i] = input;
	}

	void setDual(int i, std::vector<T> input)
	{
		this->y[i] = input;
	}
};

#endif //flexBoxDataCPU_H
