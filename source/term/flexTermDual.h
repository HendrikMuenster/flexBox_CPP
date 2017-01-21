#ifndef flexTermDual_H
#define flexTermDual_H

#include "operator/flexLinearOperator.h"
#include "prox/flexProx.h"
#include "data/flexBoxData.h"

template<typename T>
class flexTermDual
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
    std::vector<flexLinearOperator<T>* > operatorListT;

	flexProx<T>* myProx;
    std::vector<Tdata> fList;

    flexTermDual(flexProx<T>* aMyProx, T aAlpha, int numberPrimals, std::vector<flexLinearOperator<T>* > aOperatorList) : flexTermDual(aMyProx,aAlpha,numberPrimals,aOperatorList, std::vector<std::vector<T>>(0))
	{

	}

    flexTermDual(flexProx<T>* aMyProx, T aAlpha, int numberPrimals, std::vector<flexLinearOperator<T>* > aOperatorList, std::vector<std::vector<T>> _fList) : myProx(aMyProx), alpha(aAlpha), numberPrimals(numberPrimals), numberVars((int)aOperatorList.size() / numberPrimals), p(aMyProx->getProx())
    {
        fList.resize(_fList.size());

        for (int i = 0; i < fList.size(); ++i)
        {
            this->fList[i].resize(_fList[i].size());
            #ifdef __CUDACC__
                thrust::copy(_fList[i].begin(), _fList[i].end(), this->fList[i].begin());
            #else
                std::copy(_fList[i].begin(), _fList[i].end(), this->fList[i].begin());
            #endif
        }

        this->operatorList = aOperatorList;

        //create sigma and tau
        for (int i = 0; i < aOperatorList.size() / numberPrimals; ++i)
        {
            for (int j = 0; j < numberPrimals; ++j)
            {
                int opNum = i*numberPrimals + j;

                this->operatorListT.push_back(this->operatorList[opNum]->copy());
                this->operatorListT[opNum]->transpose();
            }
        }
    };

    int getNumberVars()
    {
        return numberVars;
    }

    int dualVarLength(int num)
    {
        return this->operatorList[num]->getNumRows();
    }

	~flexTermDual()
	{
		delete myProx;

        for (int i = (int)operatorList.size() - 1; i >= 0; --i)
        {
            delete operatorList[i];
            delete operatorListT[i];
        }

        operatorList.clear();
        operatorListT.clear();

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
