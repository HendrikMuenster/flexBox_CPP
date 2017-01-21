#ifndef flexProx_H
#define flexProx_H

#include <vector>

#include "tools.h"
#include "data/flexBoxData.h"

template<typename T>
class flexProx
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:
    const prox p;

	flexProx(prox _p) : p(_p)
	{

	}

	~flexProx()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

    prox getProx()
    {
        return p;
    }

	virtual void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers) = 0;

	virtual void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList) = 0;
};

#endif
