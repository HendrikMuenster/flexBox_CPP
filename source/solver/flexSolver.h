#ifndef flexSolver_H
#define flexSolver_H

#include <vector>

//! FlexBox solver class
/*!
	flexSolver is an internal abstract class for running the primal dual algortihm
	used by the flexBox main class.
	This class should not be used directly.
*/
template<typename T>
class flexSolver
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:
	flexSolver()
	{

	}

	virtual ~flexSolver()
	{
		if (VERBOSE > 0) printf("Destructor solver\n!");
	}

	virtual void init(flexBoxData<T> *data) = 0;

	virtual void addTerm(flexBoxData<T> *data, flexTerm<T>* _dualPart, std::vector<int> _correspondingPrimals) = 0;

	virtual void doIteration(flexBoxData<T> *data) = 0;

	virtual T calculateError(flexBoxData<T> *data) = 0;
};

#endif
