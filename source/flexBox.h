#ifndef flexbox_H
#define flexbox_H

#include "term/flexTerm.h"

#ifdef __CUDACC__
	#include "solver/flexSolverPrimalDualCuda.h"
	#include "data/flexBoxDataGPU.h"
	#include <device_functions.h>
#else
	#include "solver/flexSolverPrimalDual.h"
  #include "data/flexBoxDataCPU.h"
#endif

#include <vector>

//! FlexBox main class
/*!
	flexBox gathers information about the problem size, modifies the algorithm parameters, runs the
	optimization algorithm and returns the result.
*/
template <typename T>
class flexBox
{
	typedef flexLinearOperator<T> linOpType;

private:
	flexBoxData<T>* data;
	flexSolver<T>* solver;

	public:
		T tol; 							//!< stopping tolerance using the primal dual residual proposed by Goldstein, Esser and Baraniuk \cite goldstein2013adaptive. Default value is 1e-5.
		int maxIterations; 	//!< maximum number of iterations if tol has not been reached. Default value is 10000. \sa tol
		int checkError;			//!< number of steps after which to calculate the primal dual residual. Default value is 100.
		int displayStatus;	//!< number of steps after which to print status information. Default value is 1000.
		int verbose;				//!< controls the amount of information printed. Possible values are 0, 1 or 2. Default value is 0.
		bool isMATLAB; 			//!< indicates whether flexBox is used via MALTAB. Default value is false.

		std::vector<std::vector<int>> dims; //!< contains lists of dimensions of primal variables.

		//! initializes the main object setting default values for parameters. \sa tol maxIterations checkError displayStatus verbose isMATLAB
		flexBox()
		{
			this->tol = static_cast<T>(1e-5);

			this->maxIterations = static_cast<int>(10000);
			this->checkError = static_cast<int>(100);
			this->displayStatus = static_cast<int>(1000);
			this->verbose = static_cast<int>(0);

			#ifdef __CUDACC__
				this->data = new flexBoxDataGPU<T>();
				this->solver = new flexSolverPrimalDualCuda<T>();
			#else
				this->data = new flexBoxDataCPU<T>();
				this->solver = new flexSolverPrimalDual<T>();
			#endif

			this->isMATLAB = false;

		};

		~flexBox()
		{
			delete data;
			delete solver;
		}

		//! returns the number of primal vars
		/*!
			\return number of primal vars
		*/
		int getNumPrimalVars() const
		{
			return data->getNumPrimalVars();
		}

		//! returns the number of dual vars
		/*!
			\return number of dual vars
		*/
		int getNumDualVars() const
		{
			return data->getNumDualVars();
		}

		//! returns the requested primal variable
		/*!
			\param i internal identifcation returned by addPrimalVar() \sa addPrimalVar()
			\return primal variable identifed by i
		*/
		std::vector<T> getPrimal(int i)
		{
			return data->getPrimal(i);
		}

		//! returns the requested dual variable
		/*!
			\param i internal identifcation
			\return dual variable identifed by i
		*/
		std::vector<T> getDual(int i)
		{
			return data->getDual(i);
		}

		//! set the primal variable identifed by i
		/*!
			\param i internal identifcation for variable
			\param input data for primal variable
		*/
		void setPrimal(int i, std::vector<T> input)
		{
			data->setPrimal(i, input);
		}

		//! set the dual variable identifed by i
		/*!
			\param i internal identifcation for variable
			\param input data for dual variable
		*/
		void setDual(int i, std::vector<T> input)
		{
			data->setDual(i, input);
		}

		//! returns the dimensions of primal vars identified by i
		/*!
			\param i internal identifcation for variable
			\return dimension of primal variable
		*/
		std::vector<int> getDims(int i)
		{
			return dims.at(i);
		}

		//! adds a primal variable
		/*!
			\param dims dimension of variable
			\return internal identifcation for variable
		*/
		int addPrimalVar(std::vector<int> _dims)
		{
			int numberOfElements = vectorProduct(_dims);

			data->addPrimalVar(vectorProduct(_dims));

			dims.push_back(_dims);

			return getNumPrimalVars() - 1;
		}

		//! adds a Term
		/*!
			\param aTerm term to be added
			\param aPrimala corresponding primal variables for term
		*/
		void addTerm(flexTerm<T>* aTerm, std::vector<int> aPrimals)
		{
			solver->addTerm(data, aTerm, aPrimals);
		}

		//! runs the optimization algorithm
		void runAlgorithm()
		{
			solver->init(data);

			T error = static_cast<int>(1);
			int iteration = 0;

			Timer timer;
			Timer timer2;
			bool doTime = true;

			if (doTime) timer.reset();

			while (error > tol && iteration < maxIterations)
			{
				//timer2.reset();
				solver->doIteration(data);
				//timer2.end(); printf("Time for iteration was: %f\n", timer2.elapsed());

				if (iteration % displayStatus == 1)
				{
					if (this->isMATLAB)
					{
						if (this->verbose > 0)
						{
							#if IS_MATLAB
							mexPrintf("Iteration #%d | Error:%f\n", iteration, error);
							mexEvalString("pause(.0001);");
							#endif

						}
					}
					else
					{

					}
				}

				if (iteration % checkError == 0)
				{
					error = solver->calculateError(data);
				}
                //error = solver->calculateError(data);

                //printf("%f\n",error);

				++iteration;
			}

			if (doTime) timer.end();

			if (this->verbose > 0)
			{
				if (doTime) printf("Time for %d Iterations was: %f\n", iteration, timer.elapsed());
			}

		}
};

#endif
