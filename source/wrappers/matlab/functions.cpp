/*
% Author Information:
% Hendrik Dirks
% Institute for Computational and Applied Mathematics
% University of Muenster, Germany
%
% Contact: hendrik.dirks@wwu.de
%
%
% Version 1.0
% Date: 2015-06-17

% All Rights Reserved
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a
% commercial product is hereby granted without fee, provided that the
% above copyright notice appear in all copies and that both that
% copyright notice and this permission notice appear in supporting
% documentation, and that the name of the author and University of Muenster not be used in
% advertising or publicity pertaining to distribution of the software
% without specific, written prior permission.
*/

//uncomment later and compiler directive
//#define __CUDACC__ 1
//#define DO_CUDA_CHECK 1

#define IS_MATLAB 1

#include "mex.h"
#include "math.h"
#include <omp.h>
#include <iostream>

#include <stdio.h>
#include <sys/types.h>
#include <string.h>
#include <cstddef>
#include <ctime>

#include "tools.h"

#include "operator/flexLinearOperator.h"
#include "operator/flexIdentityOperator.h"
#include "operator/flexZeroOperator.h"
#include "operator/flexDiagonalOperator.h"
#include "operator/flexMatrix.h"
#include "operator/flexMatrixLogical.h"
#include "operator/flexFullMatrix.h"
#include "operator/flexGradientOperator.h"
#include "operator/flexSuperpixelOperator.h"
#include "operator/flexConcatOperator.h"

#include "flexBox.h"

#include "term/flexTerm.h"

//prox
#include "prox/flexProxDualDataL1.h"
#include "prox/flexProxDualDataL2.h"
#include "prox/flexProxDualDataKL.h"
#include "prox/flexProxDualDataHuber.h"
#include "prox/flexProxDualL1Aniso.h"
#include "prox/flexProxDualL1Iso.h"
#include "prox/flexProxDualL2.h"
#include "prox/flexProxDualL2Inf.h"
#include "prox/flexProxDualLInf.h"
#include "prox/flexProxDualHuber.h"
#include "prox/flexProxDualFrobenius.h"
#include "prox/flexProxDualBoxConstraint.h"
#include "prox/flexProxDualInnerProduct.h"
#include "prox/flexProxDualLabeling.h"

typedef double floatingType;

#ifdef __CUDACC__
	using namespace thrust;
	#include "operator/flexMatrixGPU.h"

	typedef thrust::device_vector<floatingType> vectorData;
#else
	using namespace std;
	typedef std::vector<floatingType> vectorData;
#endif


flexLinearOperator<floatingType>* transformMatlabToFlexOperator(mxArray *pointerA, int verbose, int operatorNumber);
void copyToVector(std::vector<floatingType> &vector, const double *input, int numElements);
bool checkClassType(mxArray *object, const std::string& className);
bool checkSparse(mxArray *object);
bool checkFullMatrix(mxArray *object);
bool checkProx(mxArray *inputClass, const char* proxName);

void copyMatlabToFlexFullMatrix(const mxArray *input, flexFullMatrix<floatingType> *output);
void copyMatlabToFlexmatrix(const mxArray *input, flexMatrix<floatingType> *output);
void copyMatlabToFlexLogicalMatrix(const mxArray *input, flexMatrixLogical<floatingType> *output);



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	bool isGPU = false;
    #ifdef __CUDACC__
        isGPU = true;
    #endif

// Initialize main flexBox object
	flexBox<floatingType> mainObject;
	mainObject.isMATLAB = true;

	// read params
	mxArray *params = mxGetProperty(prhs[0],0,"params");

	int numMaxIt = mxGetFieldNumber(params, "maxIt");
	if (numMaxIt >= 0)
	{
		mainObject.maxIterations = (int)mxGetScalar(mxGetFieldByNumber(params, 0, numMaxIt));
	}

	int numVerbose = mxGetFieldNumber(params, "verbose");
	if (numVerbose >= 0)
	{
		mainObject.verbose = (int)mxGetScalar(mxGetFieldByNumber(params, 0, numVerbose));
	}

	int numTol = mxGetFieldNumber(params, "tol");
	if (numTol >= 0)
	{
		mainObject.tol = (float)mxGetScalar(mxGetFieldByNumber(params, 0, numTol));
	}

	int numCheckError = mxGetFieldNumber(params, "checkError");
	if (numCheckError >= 0)
	{
		mainObject.checkError = (int)mxGetScalar(mxGetFieldByNumber(params, 0, numCheckError));
	}

    

	int verbose = mainObject.verbose;

	if (verbose > 0)
	{
		printf("Parameters:\n");

		printf("maxIterations: %d\n",mainObject.maxIterations);
		printf("verbose: %d\n",mainObject.verbose);
		printf("tol: %f\n", mainObject.tol);
		printf("checkError: %d\n", mainObject.checkError);
	}

	// read primal vars
	mxArray *x = mxGetProperty(prhs[0],0,"x");
	mxArray *dims = mxGetProperty(prhs[0],0,"dims");

	int numPrimalVars = (int)mxGetN(x)*(int)mxGetM(x);

	for (int i=0;i < numPrimalVars; ++i)
	{
		std::vector<int> _dims;

		double *input_dims = mxGetPr(mxGetCell(dims,i));

		int numberOfElementsVector = 1;
		for (int j = 0; j < mxGetN(mxGetCell(dims,i)) * mxGetM(mxGetCell(dims,i)); ++j)
		{
			_dims.push_back((int)input_dims[j]);

			numberOfElementsVector *= (int)input_dims[j];
		}

		//add primal variable
		mainObject.addPrimalVar(_dims);

		//copy matlab variable to c++ variable
		std::vector<floatingType> tmpVector(numberOfElementsVector, 0.0);

		copyToVector(tmpVector, mxGetPr(mxGetCell(x,i)), numberOfElementsVector);

		mainObject.setPrimal(i, tmpVector);
	}

    

	// copy primal terms
	mxArray *duals = mxGetProperty(prhs[0],0,"duals");
	mxArray *dcp = mxGetProperty(prhs[0],0,"DcP");  //numbers of primal variables corresponding to dual terms
	mxArray *dcd = mxGetProperty(prhs[0],0,"DcD");  //numbers of dual variables corresponding to dual terms

	int numDualTerms = (int)mxGetN(duals) * (int)mxGetM(duals);
	for (int i=0;i < numDualTerms; ++i)
	{
		mxArray* classPointer = mxGetCell(duals,i);
		const char* class_name = mxGetClassName(mxGetCell(duals,i));

		//weight
		float alpha = (float)mxGetScalar(mxGetProperty(mxGetCell(duals,i),0,"factor"));

		if (verbose > 1)
		{
			mexPrintf("Dual term %i is of type %s with alpha = %f\n",i,mxGetClassName(mxGetCell(duals,i)),alpha);
		}

		double *input_correspondingPrimals = mxGetPr( mxGetCell(dcp,i) );

		std::vector<int> _correspondingPrimals;
		for (int j = 0; j < mxGetN(mxGetCell(dcp,i)) * mxGetM(mxGetCell(dcp,i)); ++j)
		{
			//decrease number by 1 because C++ internal counter starts at 0
			_correspondingPrimals.push_back((int)input_correspondingPrimals[j] - 1);

			if (verbose > 1)
			{
				printf("Dual term #%d corresponds to primal var #%d\n",i,(int)input_correspondingPrimals[j] - 1);
			}
		}

		//create list of operators
		mxArray *matlabOperatorList = mxGetProperty(mxGetCell(duals,i),0,"operator");

		int numberOfOperators = (int)mxGetN(matlabOperatorList) * (int)mxGetM(matlabOperatorList);

		std::vector<flexLinearOperator<floatingType>*> operatorList;
		for (int k = 0; k < numberOfOperators; ++k)
		{
			int correspondingNumberPrimalVar = k%_correspondingPrimals.size();

			mxArray* pointerA = mxGetCell(matlabOperatorList,k);

			operatorList.push_back(transformMatlabToFlexOperator(pointerA, verbose, k));
		}

        mxDestroyArray(matlabOperatorList);

		flexProx<floatingType>* myProx;

		if (checkProx(classPointer,"L1IsoProxDual"))
		{
			myProx = new flexProxDualL1Iso<floatingType>();
		}
		else if (checkProx(classPointer,"L1AnisoProxDual"))
		{
			myProx = new flexProxDualL1Aniso<floatingType>();
		}
		else if (checkProx(classPointer,"L2proxDual"))
		{
			myProx = new flexProxDualL2<floatingType>();
		}
        else if (checkProx(classPointer, "L2InfProxDual"))
        {
            myProx = new flexProxDualL2Inf<floatingType>();
        }
        else if (checkProx(classPointer, "LInfProxDual"))
        {
            myProx = new flexProxDualLInf<floatingType>();
        }
		else if (checkProx(classPointer,"HuberProxDual"))
		{
			float huberEpsilon = (float)mxGetScalar(mxGetProperty(mxGetCell(duals,i),0,"epsi"));
			myProx = new flexProxDualHuber<floatingType>(huberEpsilon);
		}
		else if (checkProx(classPointer,"FrobeniusProxDual"))
		{
			myProx = new flexProxDualFrobenius<floatingType>();
		}
		//data
		else if (checkProx(classPointer,"L2DataProxDual"))
		{
			myProx = new flexProxDualDataL2<floatingType>();
		}
		else if (checkProx(classPointer,"L1DataProxDual"))
		{
			myProx = new flexProxDualDataL1<floatingType>();
		}
		else if (checkProx(classPointer,"KLDataProxDual"))
		{
			myProx = new flexProxDualDataKL<floatingType>();
		}
		else if (checkProx(classPointer,"HuberDataProxDual"))
		{
			float huberEpsilon = (float)mxGetScalar(mxGetProperty(mxGetCell(duals,i),0,"epsi"));
			myProx = new flexProxDualDataHuber<floatingType>(huberEpsilon);
		}
		else if (checkProx(classPointer,"constraintBoxDualized"))
		{
			float minVal = (float)mxGetScalar(mxGetProperty(mxGetCell(duals,i),0,"minVal"));
			float maxVal = (float)mxGetScalar(mxGetProperty(mxGetCell(duals,i),0,"maxVal"));
			myProx = new flexProxDualBoxConstraint<floatingType>(minVal, maxVal);
		}
        else if (checkProx(classPointer,"innerProductProxDual"))
		{
			myProx = new flexProxDualInnerProduct<floatingType>();
		}
		else if (checkProx(classPointer,"labelingProxDual"))
		{
			myProx = new flexProxDualLabeling<floatingType>();
		}
		else
		{
			mexPrintf("Prox not found");
			mexErrMsgTxt("Aborting");
		}

		mxArray* fListInput = mxGetProperty(mxGetCell(duals,i),0,"f");
		int sizeFList = (int)mxGetN(fListInput) * (int)mxGetM(fListInput);

		std::vector<std::vector<floatingType>> fList;
		fList.resize(sizeFList);

		for (int k = 0; k < sizeFList; ++k)
		{
			mxArray* fListInputElement = mxGetCell(fListInput,k);
			//copy elements from matlab to fList vector
			copyToVector(fList[k], mxGetPr(fListInputElement), (int)mxGetN(fListInputElement) * (int)mxGetM(fListInputElement));
		}

        mxDestroyArray(fListInput);
		mainObject.addTerm(new flexTerm<floatingType>(myProx, alpha, (int)_correspondingPrimals.size(), operatorList, fList), _correspondingPrimals);
	}

    

	// copy content for dual vars from MATLAB
	mxArray* y = mxGetProperty(prhs[0],0,"y");

	int numberDualVars = mainObject.getNumDualVars();
	for (int i=0;i < numberDualVars; ++i)
	{
        mxArray* yElement = mxGetCell(y, i);

		int numberOfElementsVector = (int)mxGetN(yElement) * (int)mxGetM(yElement);
		//copy matlab variable to c++ variable
		std::vector<floatingType> tmpVector(numberOfElementsVector, 0.0);
		copyToVector(tmpVector, mxGetPr(yElement), numberOfElementsVector);
		mainObject.setDual(i, tmpVector);
	}

    //cleanup
    mxDestroyArray(params);
    mxDestroyArray(x);
    mxDestroyArray(dims);
    
    mxDestroyArray(y);
    mxDestroyArray(duals);
    mxDestroyArray(dcd);
    mxDestroyArray(dcp);

	mainObject.runAlgorithm();



	//send content of primal vars
	//retrieve results from FlexBox
	for (int i = 0; i < numPrimalVars; ++i)
	{
		std::vector<floatingType> flexResult = mainObject.getPrimal(i);

		size_t *resultSize = new size_t[2];
		resultSize[0] = flexResult.size();
		resultSize[1] = 1;

		plhs[i] = mxCreateNumericArray(2, resultSize, mxDOUBLE_CLASS, mxREAL);
		double *ptrResult = mxGetPr(plhs[i]);
		for (int j = 0; j < resultSize[0]; ++j)
		{
			ptrResult[j] = flexResult[j];
		}

        delete[] resultSize;

	}

	//send content of dual vars
	//retrieve results from FlexBox
	for (int i = 0; i < numberDualVars; ++i)
	{
		std::vector<floatingType> flexResult = mainObject.getDual(i);

		size_t *resultSize = new size_t[2];
		resultSize[0] = flexResult.size();
		resultSize[1] = 1;

		plhs[numPrimalVars + i] = mxCreateNumericArray(2, resultSize, mxDOUBLE_CLASS, mxREAL);
		double *ptrResult = mxGetPr(plhs[numPrimalVars+i]);

		for (int j = 0; j < resultSize[0]; ++j)
		{
			ptrResult[j] = flexResult[j];
		}

        delete[] resultSize;
	}
}

flexLinearOperator<floatingType>* transformMatlabToFlexOperator(mxArray *pointerA, int verbose, int operatorNumber)
{
	flexLinearOperator<floatingType>*A;

	bool isGPU = false;
	#ifdef __CUDACC__
	isGPU = true;
	#endif

	bool isMinus = false;

	if (mxGetProperty(pointerA, 0, "isMinus") != NULL) //matrix does not have this property
	{
		isMinus = mxGetScalar(mxGetProperty(pointerA, 0, "isMinus")) > 0;
	}

	if (verbose > 1)
	{
		printf("isMinus is set to %d\n", isMinus);
	}

	if (checkClassType(pointerA, std::string("functionHandleOperator")))
	{
		mexErrMsgTxt("Operator type functionHandleOperator not supported!\n");
	}
	if (checkClassType(pointerA, std::string("gradientOperator")))
	{
		if (verbose > 1)
		{
			printf("Operator %d is type <gradientOperator>\n", operatorNumber);
		}

		char *gradientTypeString = mxArrayToString(mxGetProperty(pointerA, 0, "type"));
		int gradientDirection = static_cast<int>(mxGetScalar(mxGetProperty(pointerA, 0, "gradDirection"))) - 1; //substract one!

		gradientType gradT = gradientType::forward;
		if (strcmp(gradientTypeString, "backward") == 0)
		{
			gradT = backward;
		}
		else if (strcmp(gradientTypeString, "central") == 0)
		{
			gradT = central;
		}

		auto inputDimensionMatlab = mxGetProperty(pointerA, 0, "inputDimension");
		double *inputDimensionMatlabPtr = mxGetPr(inputDimensionMatlab);
		std::vector<int> tmpDiagonal(mxGetM(inputDimensionMatlab) * mxGetN(inputDimensionMatlab), 0);

		for (int l = 0; l < mxGetM(inputDimensionMatlab) * mxGetN(inputDimensionMatlab); ++l)
		{
			tmpDiagonal[l] = static_cast<int>(inputDimensionMatlabPtr[l]);
		}

		A = new flexGradientOperator<floatingType>(tmpDiagonal, gradientDirection, gradT, isMinus);
	}
	else if (checkClassType(pointerA, std::string("identityOperator")))
	{
		if (verbose > 1)
		{
			printf("Operator %d is type <identityOperator>\n", operatorNumber);
		}

		int nPx = static_cast<int>(mxGetScalar(mxGetProperty(pointerA, 0, "nPx")));

		A = new flexIdentityOperator<floatingType>(nPx, nPx, isMinus);
	}
	else if (checkClassType(pointerA, std::string("zeroOperator")))
	{
		if (verbose > 1)
		{
			printf("Operator %d is type <zeroOperator>\n", operatorNumber);
		}

		int nPx = static_cast<int>(mxGetScalar(mxGetProperty(pointerA, 0, "nPx")));

		A = new flexZeroOperator<floatingType>(nPx, nPx, isMinus);
	}
	else if (checkClassType(pointerA, std::string("diagonalOperator")))
	{
		if (verbose > 1)
		{
			printf("Operator %d is type <diagonalOperator>\n", operatorNumber);
		}

		//copy diagonal vector
		auto diagElements = mxGetProperty(pointerA, 0, "diagonalElements");
		double *tmpDiagonalVector = mxGetPr(diagElements);
		std::vector<floatingType> tmpDiagonal(mxGetM(diagElements) * mxGetN(diagElements), static_cast<floatingType>(0));

		for (int l = 0; l < mxGetM(diagElements) * mxGetN(diagElements); ++l)
		{
			tmpDiagonal[l] = static_cast<floatingType>(tmpDiagonalVector[l]);
		}

		A = new flexDiagonalOperator<floatingType>(tmpDiagonal, isMinus);
	}
	else if (checkClassType(pointerA, std::string("concatOperator")))
	{
		if (verbose > 1)
		{
			printf("Operator %d is type <concatOperator>\n", operatorNumber);
		}

		std::string operationMatlab = mxArrayToString(mxGetProperty(pointerA, 0, "operation"));

		mySign operation;

		if (operationMatlab.compare("composition") == 0)
		{
			operation = COMPOSE;
		}
		else if (operationMatlab.compare("addition") == 0)
		{
			operation = PLUS;
		}
		else if (operationMatlab.compare("difference") == 0)
		{
			operation = MINUS;
		}
		else
		{
			mexErrMsgTxt("Unknown operation for concatOperator\n");
		}

		auto operator1 = transformMatlabToFlexOperator(mxGetProperty(pointerA, 0, "A"), verbose, operatorNumber);
		auto operator2 = transformMatlabToFlexOperator(mxGetProperty(pointerA, 0, "B"), verbose, operatorNumber);

		A = new flexConcatOperator<floatingType>(operator1, operator2, operation, isMinus);
	}
	else if (checkClassType(pointerA, std::string("superpixelOperator")) && isGPU == false)
	{
		if (verbose > 1)
		{
			printf("Operator %d is type <superpixelOperator>\n", operatorNumber);
		}

		float factor = (float)mxGetScalar(mxGetProperty(pointerA, 0, "factor"));// factor that f is being upsized
		//dimension of data f

		auto targetDimensionStruct = mxGetProperty(pointerA, 0, "targetDimension");
		double *targetDimensionInput = mxGetPr(targetDimensionStruct);
		int targetDimensionSize = (int)(mxGetN(targetDimensionStruct) * mxGetM(targetDimensionStruct));
		std::vector<int> targetDimension(targetDimensionSize, 0);
		for (int l = 0; l < targetDimensionSize; ++l)
		{
			targetDimension[l] = (int)targetDimensionInput[l];
		}

		A = new flexSuperpixelOperator<floatingType>(targetDimension, factor, isMinus);
	}
	else if (checkSparse(pointerA) || (checkClassType(pointerA, std::string("superpixelOperator")) && isGPU == true))
	{
		if (verbose > 1)
		{
			printf("Operator %d is type <matrix>\n", operatorNumber);
		}

		//check if super pixel operator
		if (checkClassType(pointerA, std::string("superpixelOperator")))
		{
			pointerA = mxGetProperty(pointerA, 0, "matrix");
		}

		#ifdef __CUDACC__
		mwIndex  *ir, *jc;

		jc = mxGetJc(pointerA);
		ir = mxGetIr(pointerA);
		double* pr = mxGetPr(pointerA);

		//matlab stores in compressed column format
		int numCols = mxGetN(pointerA);
		int* colList = new int[numCols + 1];
		for (int l = 0; l <= numCols; ++l)
		{
			colList[l] = jc[l];
		}

		int nnz = colList[numCols];

		int* rowList = new int[nnz];
		float* valList = new float[nnz];
		for (int l = 0; l < nnz; ++l)
		{
			rowList[l] = ir[l];
			valList[l] = pr[l];
		}
		A = new flexMatrixGPU<floatingType>((int)mxGetM(pointerA), (int)mxGetN(pointerA), rowList, colList, valList, false, isMinus);
        delete[] colList; delete[] rowList; delete[] valList;
		#else
		auto Atmp = new flexMatrix<floatingType>(static_cast<int>(mxGetM(pointerA)), static_cast<int>(mxGetN(pointerA)), isMinus);
		copyMatlabToFlexmatrix(pointerA, Atmp);

		A = Atmp;
		#endif
	}
	else if (mxIsLogical(pointerA))
	{
		auto Atmp = new flexMatrixLogical<floatingType>((int)mxGetM(pointerA), (int)mxGetN(pointerA), isMinus);

		copyMatlabToFlexLogicalMatrix(pointerA, Atmp);

		A = Atmp;
	}
	else if (checkFullMatrix(pointerA))
	{
		auto Atmp = new flexFullMatrix<floatingType>((int)mxGetM(pointerA), (int)mxGetN(pointerA), isMinus);

		copyMatlabToFlexFullMatrix(pointerA, Atmp);

		A = Atmp;
	}
	else
	{
		mexErrMsgTxt("Operator type not supported!\n");
	}

	return A;
}



void copyToVector(std::vector<floatingType> &vector,const double *input, int numElements)
{
	//resize target vector
	vector.resize(numElements);

	for (int j = 0; j < numElements; ++j)
	{
		vector[j] = (floatingType)input[j];
	}
}

bool checkClassType(mxArray *object, const std::string& className)
{
	mxArray *output[1], *input[2];

	input[0] = object;
	input[1] = mxCreateString(className.c_str());

	mexCallMATLAB(1, output, 2, input, "isa");

	if (mxGetScalar(output[0]) > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool checkSparse(mxArray *object)
{
	mxArray *output[1], *input[1];

	input[0] = object;

	mexCallMATLAB(1, output, 1, input, "issparse");

	if (mxGetScalar(output[0]) > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool checkFullMatrix(mxArray *object)
{
	mxArray *output[1], *input[1];

	input[0] = object;

	mexCallMATLAB(1, output, 1, input, "ismatrix");

	if (mxGetScalar(output[0]) > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool checkProx(mxArray *inputClass, const char* proxName)
{
	mxArray *output[1], *input[1];

	input[0] = inputClass;

	mexCallMATLAB(1, output, 1, input, "superclasses");

	for (int j = 0; j < mxGetN(output[0]) * mxGetM(output[0]); ++j)
	{
		const char* class_name = mxArrayToString(mxGetCell(output[0],j));

		if (strcmp(class_name, proxName) == 0)
		{
			return true;
		}
	}

	return false;
}

void copyMatlabToFlexFullMatrix(const mxArray *input, flexFullMatrix<floatingType> *output)
{
	double* values = mxGetPr(input);

	for (int i = 0; i < mxGetM(input)*mxGetN(input); ++i)
	{
		output->insertElement(i, values[i]);
	}
}

void copyMatlabToFlexLogicalMatrix(const mxArray *input, flexMatrixLogical<floatingType> *output)
{
	bool* values = mxGetLogicals(input);

	/*std::vector<int> indexI(0, 0);
	std::vector<int> indexJ(0, 0);

	for (int j = 0; j < mxGetN(input); ++j)
	{
		for (int i = 0; i < mxGetM(input); ++i)
		{
			if (values[i + j*mxGetM(input)])
			{
				indexI.push_back(i);
				indexJ.push_back(j);
			}
		}
	}

	output->blockInsert(indexI, indexJ);*/

	int sizeM = static_cast<int>(mxGetM(input));
	int sizeN = static_cast<int>(mxGetN(input));

	for (int i = 0; i < sizeM; ++i)
	{
		int numElements = 0;
		for (int j = 0; j < sizeN; ++j)
		{
			if (values[i + j*sizeM])
			{
				output->indexList.push_back(j);
				++numElements;
			}
		}
		output->rowToIndexList[i + 1] = output->rowToIndexList[i] + numElements;
	}
}

void copyMatlabToFlexmatrix(const mxArray *input, flexMatrix<floatingType> *output)
{
	double  *pr;
	mwIndex  *ir, *jc;
	mwSize      col, total = 0;
	mwIndex   starting_row_index, stopping_row_index, current_row_index;
	mwSize      n;

	std::vector<int> indexI(0, 0);
	std::vector<int> indexJ(0, 0);
	std::vector<floatingType> indexVal(0, 0.0f);

	pr = mxGetPr(input);
	ir = mxGetIr(input);
	jc = mxGetJc(input);

	n = mxGetN(input);
	for (col = 0; col<n; col++)
	{
		starting_row_index = jc[col];
		stopping_row_index = jc[col + 1];
		if (starting_row_index == stopping_row_index)
			continue;
		else
		{
			for (current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++)
			{
				indexI.push_back(static_cast<int>(ir[current_row_index]));
				indexJ.push_back(static_cast<int>(col));
				indexVal.push_back(static_cast<floatingType>(pr[total]));

				total++;
			}
		}
	}

	output->blockInsert(indexI, indexJ, indexVal);
}
