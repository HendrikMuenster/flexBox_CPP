#ifndef flexTerm_H
#define flexTerm_H

#include "operator/flexLinearOperator.h"
#include "prox/flexProx.h"
#include "data/flexBoxData.h"

//! wrapper class for all usable terms
/*!
	flexTerm is a wrapper class for all terms, setting the correct proximal, storing
	the needed operators and maintaining the used variables.
*/
template<typename T>
class flexTerm
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
    int numberVars;
    int numberPrimals;
public:
    const prox p;
    T alpha;
    std::vector<flexLinearOperator<T>* > operatorList;

	flexProx<T>* myProx;
    std::vector<Tdata> fList;

	flexTerm(flexProx<T>* aMyProx, T aAlpha, int numberPrimals, std::vector<flexLinearOperator<T>* > aOperatorList) : flexTerm(aMyProx, aAlpha, numberPrimals, aOperatorList, std::vector<std::vector<T>>(0))
	{

	}

	flexTerm(flexProx<T>* aMyProx, T aAlpha, int numberPrimals, std::vector<flexLinearOperator<T>* > aOperatorList, std::vector<std::vector<T>> aFList) : myProx(aMyProx), alpha(aAlpha), numberPrimals(numberPrimals), numberVars((int)aOperatorList.size() / numberPrimals), p(aMyProx->getProx())
    {
        fList.resize(aFList.size());

        for (int i = 0; i < fList.size(); ++i)
        {
            this->fList[i].resize(aFList[i].size());
            #ifdef __CUDACC__
                thrust::copy(aFList[i].begin(), aFList[i].end(), this->fList[i].begin());
            #else
                std::copy(aFList[i].begin(), aFList[i].end(), this->fList[i].begin());
            #endif
        }

        this->operatorList = aOperatorList;
    }

    int getNumberVars()
    {
        return numberVars;
    }

    int dualVarLength(int num)
    {
        return this->operatorList[num]->getNumRows();
    }

	~flexTerm()
	{
		delete myProx;

        for (int i = (int)operatorList.size() - 1; i >= 0; --i)
        {
            delete operatorList[i];
        }

        operatorList.clear();

		if (VERBOSE > 0) printf("Destructor of data term!");
	}

	void applyProx(flexBoxData<T>* data, const std::vector<int> &dualNumbers, const std::vector<int> &primalNumbers)
	{
        //apply both prox operators. Usually one is empty
        myProx->applyProx(this->alpha, data,dualNumbers,primalNumbers,this->fList);
        myProx->applyProx(this->alpha, data,dualNumbers,primalNumbers);
	};
};

#endif
