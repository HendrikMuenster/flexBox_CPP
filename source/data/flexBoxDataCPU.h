#ifndef flexBoxDataCPU_H
#define flexBoxDataCPU_H

#include "flexBoxData.h"

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
