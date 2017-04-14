#ifndef flexDiagonalOperator_H
#define flexDiagonalOperator_H

#include <vector>
#include "flexLinearOperator.h"

//! represents a diagonal operator
template <typename T>
class flexDiagonalOperator : public flexLinearOperator<T>
{

#ifdef __CUDACC__
	typedef thrust::device_vector<T> Tdata;
#else
	typedef std::vector<T> Tdata;
#endif

private:
	Tdata diagonalElements;
public:

	//! initializes the concatenation operator for non-CUDA versions
	/*!
		\param aDiagonalElements vector of diagonal Elements
		\param aMinus determines if operator is negated \sa isMinus
	*/
	flexDiagonalOperator(std::vector<T> aDiagonalElements, bool aMinus)
	: flexLinearOperator<T>(static_cast<int>(aDiagonalElements.size()), static_cast<int>(aDiagonalElements.size()), diagonalOp, aMinus)
	{
		this->diagonalElements.resize(aDiagonalElements.size());

		#ifdef __CUDACC__
			thrust::copy(aDiagonalElements.begin(), aDiagonalElements.end(), this->diagonalElements.begin());

		#else
			this->diagonalElements = aDiagonalElements;
		#endif
	}

	#ifdef __CUDACC__
		//! initializes the concatenation operator for CUDA versions
		/*!
			\param aDiagonalElements vector of diagonal Elements where Tdata is of type thrust::device_vector<T>
			\param aMinus determines if operator is negated \sa isMinus
		*/
		flexDiagonalOperator(Tdata aDiagonalElements, bool aMinus) : diagonalElements(aDiagonalElements), flexLinearOperator<T>(static_cast<int>(aDiagonalElements.size()), static_cast<int>(aDiagonalElements.size()), diagonalOp, aMinus)
		{

		};
	#endif

	flexDiagonalOperator<T>* copy()
	{
		flexDiagonalOperator<T>* A = new flexDiagonalOperator<T>(this->diagonalElements, this->isMinus);

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





	//apply linear operator to vector
	void times(bool transposed, const Tdata &input, Tdata &output)
	{
		this->doTimes(input,output,EQUALS);
	}

	void timesPlus(bool transposed, const Tdata &input, Tdata &output)
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

	void timesMinus(bool transposed, const Tdata &input, Tdata &output)
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
		Tdata diagonalElementsCopy = this->diagonalElements;

		vectorAbs(diagonalElementsCopy);

		return vectorMax(diagonalElementsCopy);
	}


	#ifdef __CUDACC__
	thrust::device_vector<T> getAbsRowSumCUDA(bool transposed)
	{
		Tdata diagonalElementsCopy = this->diagonalElements;

		vectorAbs(diagonalElementsCopy);

		return diagonalElementsCopy;
	}
	#endif

private:
	void doTimesCPU(const Tdata &input, Tdata &output,const mySign s)
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

	void doTimes(const Tdata &input, Tdata &output,const mySign s)
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
};

#endif
