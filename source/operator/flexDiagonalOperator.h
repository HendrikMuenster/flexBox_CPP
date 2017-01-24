#ifndef flexDiagonalOperator_H
#define flexDiagonalOperator_H

#include "vector"
#include "flexLinearOperator.h"

template < typename T, typename Tvector >
class flexDiagonalOperator : public flexLinearOperator<T, Tvector>
{
private:
	Tvector diagonalElements;
public:

	flexDiagonalOperator(std::vector<T> _diagonalElements, bool _minus) : flexLinearOperator<T, Tvector>((int)_diagonalElements.size(), (int)_diagonalElements.size(), diagonalOp, _minus)
	{
		this->diagonalElements.resize((int)_diagonalElements.size());

		#ifdef __CUDACC__
			thrust::copy(_diagonalElements.begin(), _diagonalElements.end(), this->diagonalElements.begin());

		#else
			this->diagonalElements = _diagonalElements;
		#endif
	};

	#ifdef __CUDACC__
	flexDiagonalOperator(Tvector _diagonalElements, bool _minus) : diagonalElements(_diagonalElements), flexLinearOperator<T, Tvector>((int)_diagonalElements.size(), (int)_diagonalElements.size(), diagonalOp, _minus){};
	#endif

	flexDiagonalOperator<T, Tvector>* copy()
	{
		flexDiagonalOperator<T, Tvector>* A = new flexDiagonalOperator<T, Tvector>(this->diagonalElements, this->isMinus);

		return A;
	}
    
    #ifdef __CUDACC__
    struct flexDiagonalOperatorFunctor
	{
		__host__ __device__ 
		flexDiagonalOperatorFunctor(const mySign _s) : s(_s){}

		template <typename Tuple>
		__host__ __device__
		void operator()(Tuple t)
		{
            switch (this->s)
            {
                case PLUS:
                {
                    thrust::get<0>(t) += thrust::get<1>(t) * thrust::get<2>(t);
                    break;
                }
                case MINUS:
                {
                    thrust::get<0>(t) -= thrust::get<1>(t) * thrust::get<2>(t);
                    break;
                }
                case EQUALS:
                {
                    thrust::get<0>(t) = thrust::get<1>(t) * thrust::get<2>(t);
                    break;
                }
            }
		}
        
        mySign s;
	};
    #endif
    
    void doTimesCPU(const Tvector &input, Tvector &output,const mySign s)
	{
        int numElements = (int)output.size();
		
		#pragma omp parallel for
		for (int i = 0; i < numElements; ++i)
		{
			switch (s)
			{
				case PLUS:
				{
					output[i] += input[i] * this->diagonalElements[i];
					break;
				}
				case MINUS:
				{
					output[i] -= input[i] * this->diagonalElements[i];
					break;
				}
				case EQUALS:
				{
					output[i] = input[i] * this->diagonalElements[i];
					break;
				}
			}
		}
    }
    
    void doTimes(const Tvector &input, Tvector &output,const mySign s)
	{
        #ifdef __CUDACC__
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(output.begin(), input.begin(), this->diagonalElements.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(output.end(),   input.end(),   this->diagonalElements.end())),
			flexDiagonalOperatorFunctor(s));
        #else
            this->doTimesCPU(input,output,s);
        #endif
    }
    
	//apply linear operator to vector
	void times(bool transposed, const Tvector &input, Tvector &output)
	{
        this->doTimes(input,output,EQUALS);
	}
    
    void timesMinus(bool transposed, const Tvector &input, Tvector &output)
	{
        if (this->isMinus)
        {
            this->doTimes(input,output, PLUS);
        }
        else
        {
            this->doTimes(input,output, MINUS);
        }
    }
    
    void timesPlus(bool transposed, const Tvector &input, Tvector &output)
	{
        if (this->isMinus)
        {
            this->doTimes(input,output, MINUS);
        }
        else
        {
            this->doTimes(input,output, PLUS);
        }
    }

	std::vector<T> getAbsRowSum(bool transposed)
	{
		std::vector<T> result(this->getNumRows());
		#pragma omp parallel for
		for (int k = 0; k < this->getNumRows(); ++k)
		{
			result[k] = std::abs(this->diagonalElements[k]);
		}

		return result;
	}

	T getMaxRowSumAbs(bool transposed)
	{
		Tvector diagonalElementsCopy = this->diagonalElements;

		vectorAbs(diagonalElementsCopy);

		return vectorMax(diagonalElementsCopy);
	}

	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		Tvector diagonalElementsCopy = this->diagonalElements;

		vectorAbs(diagonalElementsCopy);

		return diagonalElementsCopy;
	}
	#endif
};

#endif
