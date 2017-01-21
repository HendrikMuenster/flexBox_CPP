#ifndef flexBoxData_H
#define flexBoxData_H

#include <vector>

#ifdef __CUDACC__
	#include <thrust/host_vector.h>
	#include <thrust/device_vector.h>
#endif


template <typename T>
class flexBoxData
{
	#ifdef __CUDACC__
		typedef thrust::device_vector<T> Tdata;
	#else
		typedef std::vector<T> Tdata;
	#endif

	public:
	//List of primal variables
	std::vector<Tdata> x, xTmp, xOld, xTilde, xBar, xError;
	//List of dual variables
	std::vector<Tdata> y, yTmp, yOld, yTilde, yError;
	//weights
	std::vector<Tdata> tauElt;
	std::vector<Tdata> sigmaElt;

	flexBoxData(){}


	void addPrimalVar(int numberOfElements)
	{
		Tdata emptyX(numberOfElements, static_cast<T>(0));

		this->x.push_back(emptyX);
		this->xTmp.push_back(emptyX);
		this->xOld.push_back(emptyX);
		this->xBar.push_back(emptyX);
		this->xTilde.push_back(emptyX);
		this->xError.push_back(emptyX);

		this->tauElt.push_back(emptyX);

	}
	void addDualVar(int numberOfElements)
	{
		Tdata emptyY(numberOfElements, static_cast<T>(0));

		this->y.push_back(emptyY);
		this->yTmp.push_back(emptyY);
		this->yOld.push_back(emptyY);
		this->yTilde.push_back(emptyY);
		this->yError.push_back(emptyY);

		this->sigmaElt.push_back(emptyY);

	}

	int getNumPrimalVars()
	{
		return static_cast<int>(this->x.size());
	}
	int getNumDualVars()
	{
		return static_cast<int>(this->y.size());
	}

	virtual std::vector<T> getPrimal(int i) = 0;
	virtual std::vector<T> getDual(int i) = 0;

	virtual void setPrimal(int i, std::vector<T> input) = 0;
	virtual void setDual(int i, std::vector<T> input) = 0;
};

#endif
