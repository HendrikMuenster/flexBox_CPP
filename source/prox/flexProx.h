#ifndef flexProx_H
#define flexProx_H

#include <vector>

#include "tools.h"
#include "data/flexBoxData.h"

//! abstract base class for all proximals (prox)
/*!
	flexProx combines the interface for all usable proximals (prox)
*/
template<typename T>
class flexProx
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

public:
  const prox p; //!< type of prox \sa prox

	//! initializes the prox
	/*!
		\param aP type of prox
	*/
	flexProx(prox aP) : p(aP)
	{

	}

	~flexProx()
	{
		if (VERBOSE > 0) printf("Destructor prox\n!");
	}

	//! returns the type of prox
	/*!
		\return type of prox
	*/
  prox getProx()
  {
      return p;
  }

	//! applies prox for non-data terms
	/*!
		the function body should be empty if implemented prox is a data prox
		\param alpha weight of term
		\param data data object
		\param dualNumbers vector of internal identifactions of dual numbers corresponding to the term \sa flexBox
		\param primalNumbers vector of internal identifactions of primal numbers corresponding to the term \sa flexBox
	*/
	virtual void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers) = 0;

	//! applies prox for data terms
	/*!
		the function body should be empty if implemented prox is a non-data prox
		\param alpha weight of term
		\param data data object
		\param dualNumbers vector of internal identifactions of dual numbers corresponding to the term \sa flexBox
		\param primalNumbers vector of internal identifactions of primal numbers corresponding to the term \sa flexBox
		\param fList data part of term
	*/
	virtual void applyProx(T alpha, flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers, std::vector<Tdata> &fList) = 0;
};

#endif
