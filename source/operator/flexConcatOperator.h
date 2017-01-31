#ifndef flexConcatOperator_H
#define flexConcatOperator_H


#include "vector"
#include "tools.h"
#include "flexLinearOperator.h"

template<typename T>
class flexConcatOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif
    
private:
    flexLinearOperator<T>* A;
    flexLinearOperator<T>* B;
    mySign s;
	Tdata tmpVec1;
	Tdata tmpVec2;

public:

	flexConcatOperator(flexLinearOperator<T>* _A, flexLinearOperator<T>* _B, mySign _s, bool _minus) : A(_A), B(_B), s(_s), tmpVec1(_A->getNumRows()), tmpVec2(_A->getNumCols()), flexLinearOperator<T>(_A->getNumRows(), _B->getNumCols(), concatOp, _minus)
	{

	}

	flexConcatOperator<T>* copy()
	{
		auto cpOp = new flexConcatOperator<T>(this->A, this->B, this->s, this->isMinus);

		return cpOp;
	}

	//to implement
	void times(bool transposed, const Tdata &input, Tdata &output)
	{
	}
    
	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
	{
		//printf("OpTP\n");
        switch (this->s)
        {
            case PLUS:
            {
                if (this->isMinus)
                {
                    A->timesMinus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                else
                {
                    A->timesPlus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                break;
            }
            case MINUS:
            {
                if (this->isMinus)
                {
                    A->timesMinus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                else
                {
                    A->timesPlus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                break;
            }
            case COMPOSE:
            {
                if (transposed)
                {
                    //apply A first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec2.begin(), this->tmpVec2.end(), (T)0);
					#else if
						std::fill(this->tmpVec2.begin(), this->tmpVec2.end(), (T)0);
					#endif

					A->timesPlus(true, input, this->tmpVec2);
                    if (this->isMinus)
                    {
						B->timesMinus(true, this->tmpVec2, output);
                    }
                    else
                    {
						B->timesPlus(true, this->tmpVec2, output);
                    }
                }
                else
                {
                    //apply B first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec2.begin(), this->tmpVec2.end(), (T)0);
					#else if
						std::fill(this->tmpVec2.begin(), this->tmpVec2.end(), (T)0);
					#endif

					B->timesPlus(false, input, this->tmpVec2);
                    if (this->isMinus)
                    {
						A->timesMinus(false, this->tmpVec2, output);
                    }
                    else
                    {
						A->timesPlus(false, this->tmpVec2, output);
                    }
                }
                break;
            }
        }
	}

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
	{
		//printf("OmTP\n");
        switch (this->s)
        {
            case PLUS:
            {
                if (this->isMinus)
                {
                    A->timesPlus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                else
                {
                    A->timesMinus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                break;
            }
            case MINUS:
            {
                if (this->isMinus)
                {
                    A->timesPlus(transposed, input, output);
                    B->timesMinus(transposed, input, output);
                }
                else
                {
                    A->timesMinus(transposed, input, output);
                    B->timesPlus(transposed, input, output);
                }
                break;
            }
            case COMPOSE:
            {
                if (transposed)
                {
                    //apply A first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec2.begin(), this->tmpVec2.end(), (T)0);
					#else if
						std::fill(this->tmpVec2.begin(), this->tmpVec2.end(), (T)0);
					#endif
					
					A->timesPlus(true, input, tmpVec2);
                    if (this->isMinus)
                    {
						B->timesPlus(true, this->tmpVec2, output);
                    }
                    else
                    {
						B->timesMinus(true, this->tmpVec2, output);
                    }
                }
                else
                {
                    //apply B first
					#ifdef __CUDACC__
						thrust::fill(this->tmpVec1.begin(), this->tmpVec1.end(), (T)0);
					#else if
						std::fill(this->tmpVec2.begin(), this->tmpVec2.end(), (T)0);
					#endif

					B->timesPlus(false, input, this->tmpVec2);
                    if (this->isMinus)
                    {
						A->timesPlus(false, this->tmpVec2, output);
                    }
                    else
                    {
						A->timesMinus(false, this->tmpVec2, output);
                    }
                }
                break;
            }
        }
	}

    //todo
	T getMaxRowSumAbs(bool transposed)
	{
		return static_cast<T>(1);
	}

	std::vector<T> getAbsRowSum(bool transposed)
	{
        std::vector<T> result;
        
        auto rowSumA = A->getAbsRowSum(transposed);
        auto rowSumB = A->getAbsRowSum(transposed);
        
        switch (this->s)
        {
            case PLUS:
				result.resize(rowSumA.size());

				#pragma omp parallel for
				for (int k = 0; k < result.size(); ++k)
				{
					result[k] = rowSumA[k] + rowSumB[k];
				}
				break;
            case MINUS:
            {
                result.resize(rowSumA.size());
                
                #pragma omp parallel for
                for (int k = 0; k < result.size(); ++k)
                {
                    result[k] = rowSumA[k] + rowSumB[k];
                }
                break;
            }
            case COMPOSE:
            {
                T maxA = *std::max_element(rowSumA.begin(), rowSumA.end());
                T maxB = *std::max_element(rowSumB.begin(), rowSumB.end());
                T maxProd = maxA * maxB;
                
                switch (transposed)
                {
                    case true:
                    {
                        result.resize(this->B->getNumCols(), maxProd);
                        break;
                    }
                    case false:
                    {
                        result.resize(this->A->getNumRows(), maxProd);
                        break;
                    }
                }
                break;
            }
        }
        
		return result;
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
        Tdata result;
        
        auto rowSumA = A->getAbsRowSum(transposed);
        auto rowSumB = A->getAbsRowSum(transposed);
        
        switch (this->s)
        {
            case PLUS:
            case MINUS:
            {
                result.resize(rowSumA.size());
                
                #pragma omp parallel for
                for (int k = 0; k < result.size(); ++k)
                {
                    result[k] = rowSumA[k] + rowSumB[k];
                }
                break;
            }
            case COMPOSE:
            {
                T maxA = *thrust::max_element(rowSumA.begin(), rowSumA.end());
                T maxB = *thrust::max_element(rowSumB.begin(), rowSumB.end());
                T maxProd = maxA * maxB;
                
                switch (transposed)
                {
                    case true:
                    {
                        result.resize(this->B->getNumCols(), maxProd);
                        break;
                    }
                    case false:
                    {
                        result.resize(this->A->getNumRows(), maxProd);
                        break;
                    }
                }
                break;
            }
        }

		return result;
	}
	#endif
};

#endif
