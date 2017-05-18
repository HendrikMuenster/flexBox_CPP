#include <vector>
#include <iostream>
//catch
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexLinearOperator.h"
#include "flexFullMatrix.h"
#include "flexIdentityOperator.h"
#include "flexGradientOperator.h"
#include "flexProxDualDataL2.h"
#include "flexProxDualL1Iso.h"

using namespace std;

typedef double floatingType;



TEST_CASE("Operator flexFullMatrix<floatingType>", "[flexFullMatrix]") {
	floatingType tol = 1e-7;

	flexFullMatrix<floatingType> A(5, 3, false);

	//add elements to matrix. Structure is:
	/*A= [5.0 1.0 6.3;
	 1.21 65.1 -0.1;
	 3.2 -12.0 0.01;
	 -10.0 1.0 5.0;
	 -4.0 3.0 1.115]*/
	A.insertElement(0, 0, (floatingType)5.0);
	A.insertElement(0, 1, (floatingType)1.0);
	A.insertElement(0, 2, (floatingType)6.3);
	A.insertElement(1, 0, (floatingType)1.21);
	A.insertElement(1, 1, (floatingType)65.1);
	A.insertElement(1, 2, (floatingType)-0.1);
	A.insertElement(2, 0, (floatingType)3.2);
	A.insertElement(2, 1, (floatingType)-12.0);
	A.insertElement(2, 2, (floatingType)0.01);
	A.insertElement(3, 0, (floatingType)-10.0);
	A.insertElement(3, 1, (floatingType)1.0);
	A.insertElement(3, 2, (floatingType)5.0);
	A.insertElement(4, 0, (floatingType)-4.0);
	A.insertElement(4, 1, (floatingType)3.0);
	A.insertElement(4, 2, (floatingType)1.115);

	//define vector v = [-1.012, 2.7, 500.0];
	std::vector<floatingType> v = { (floatingType)-1.012, (floatingType)2.7, (floatingType)500 };

	REQUIRE(A.getNumRows() == 5);
	REQUIRE(A.getNumCols() == 3);
	REQUIRE(A.isMinus == false);

	SECTION("calculate row and col sums") 
	{
		std::vector<floatingType> rowsSums = A.getAbsRowSum(false);
		std::vector<floatingType> colsSums = A.getAbsRowSum(true);

		std::vector<floatingType> rowSumsExpected = { (floatingType)12.3, (floatingType)66.41, (floatingType)15.21, (floatingType)16, (floatingType)8.115 };
		std::vector<floatingType> colSumsExpected = { (floatingType)23.41, (floatingType)82.1, (floatingType)12.525};

		for (int i = 0; i < rowSumsExpected.size(); ++i)
			REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

		for (int i = 0; i < colSumsExpected.size(); ++i)
			REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
	}

	SECTION("check matrix vector multiplication")
	{
		std::vector<floatingType> v1 = { 12.349, -125.12, 0.812 };
		std::vector<floatingType> v2 = { 0.12, -34.744, 3.11, 0.00043, -0.003412 };

		std::vector<floatingType> resultAv1(5, (floatingType)0);
		std::vector<floatingType> resultAtv2(3, (floatingType)0);

		A.timesPlus(false,v1, resultAv1);
		A.timesPlus(true, v2, resultAtv2);

		std::vector<floatingType> resultAv1Expected = { (floatingType)-58.2594, (floatingType)-8130.45091, (floatingType)1540.96492, (floatingType)-244.55, (floatingType)-423.85062 };
		std::vector<floatingType> resultAtv2Expected = { (floatingType)-31.478892, (floatingType)-2299.044206, (floatingType)4.2598456};

		for (int i = 0; i < resultAv1Expected.size(); ++i)
			REQUIRE(std::abs(resultAv1[i] - resultAv1Expected[i]) < tol);
		for (int i = 0; i < resultAtv2Expected.size(); ++i)
			REQUIRE(std::abs(resultAtv2[i] - resultAtv2Expected[i]) < tol);
	}
}

/*int main()
{
	/*float weightDataTerm = 1.0f;
	float weightRegularizer = 0.1f;

	flexBox<floatingType> mainObject;
	mainObject.verbose = 1;

	//read original image
	CImg<unsigned char> imageOriginal(sampleFiles.imgTest);

	CImg<unsigned char>	imageOriginalGray(imageOriginal.width(), imageOriginal.height(), 1, 1, 0);
	//convert to gray using luminosity method 
	for (int i = 0; i < imageOriginal.height(); ++i)
	{
		for (int j = 0; j < imageOriginal.width(); ++j)
		{
			unsigned char R = imageOriginal(i, j, 0, 0);
			unsigned char G = imageOriginal(i, j, 0, 1);
			unsigned char B = imageOriginal(i, j, 0, 2);

			unsigned char grayValue = static_cast<unsigned char>(0.299*R + 0.587*G + 0.114*B);

			imageOriginalGray(i,j) = grayValue;
		}
	}


	CImg<unsigned char> imageNoise = imageOriginalGray;
	imageNoise.noise(20);

	//add primal variable
	std::vector<int> dims;
	dims.push_back(imageNoise.height());
	dims.push_back(imageNoise.width());
	mainObject.addPrimalVar(dims);

	//number of elements in vector
	int nPx = imageNoise.height() * imageNoise.width();

	//add empty term for primal var, because we want to solve the fully dualized problem
	std::vector<int> correspondingPrimals;
	correspondingPrimals.push_back(0);

	//add dualized data term:
	std::vector<flexLinearOperator<floatingType>*> operatorList;
	operatorList.push_back(new flexIdentityOperator<floatingType>(nPx, nPx, false));

	//reshape image to vector and normalize to [0, 1]
	std::vector<floatingType> f(nPx, 0.0f);
	for (int i = 0; i < imageNoise.height(); ++i)
	{
		for (int j = 0; j < imageNoise.width(); ++j)
		{
			f[i*imageNoise.width() + j] = static_cast<floatingType>(imageNoise[i*imageOriginal.width() + j]) / static_cast<floatingType>(255);
		}
	}

	std::vector<std::vector<floatingType>> fList = std::vector<std::vector<floatingType>>();
	fList.push_back(f);

	flexProx<floatingType>* myProx = new flexProxDualDataL2<floatingType>();
	mainObject.addTerm(new flexTerm<floatingType>(myProx, weightDataTerm, static_cast<int>(correspondingPrimals.size()), operatorList, fList), correspondingPrimals);

	//add dualized regularizer
	operatorList.clear();
	//add gradient for x and y direction as operators
	operatorList.push_back(new flexGradientOperator<floatingType>(mainObject.getDims(0), 0, gradientType::forward, false));
	operatorList.push_back(new flexGradientOperator<floatingType>(mainObject.getDims(0), 1, gradientType::forward, false));

	flexProx<floatingType>* myProx2 = new flexProxDualL1Iso<floatingType>();
	mainObject.addTerm(new flexTerm<floatingType>(myProx2, weightRegularizer, 1, operatorList), correspondingPrimals);

	mainObject.runAlgorithm();

	std::vector<float> flexResult = mainObject.getPrimal(0);

	CImg<unsigned char>	imageResult(imageNoise.width(), imageNoise.height(), 1, 1, 0);
	for (int i = 0; i < imageNoise.height(); ++i)
	{
		for (int j = 0; j < imageNoise.width(); ++j)
		{
			imageResult(i, j, 0, 0) = static_cast<unsigned char>(flexResult[j*imageNoise.width() + i] * 255);
		}
	}

	CImgDisplay main_disp(imageOriginalGray, "Original"), draw_dispGr(imageNoise, "Noise"), draw_dispR(imageResult, "Result");
	while (!main_disp.is_closed() && !draw_dispGr.is_closed() && !main_disp.is_keyESC() && !draw_dispGr.is_keyESC() && !main_disp.is_keyQ() && !draw_dispGr.is_keyQ())
	{
		// Handle display window resizing (if any)
		if (main_disp.is_resized()) main_disp.resize().display(imageOriginalGray);
		if (draw_dispGr.is_resized()) draw_dispGr.resize().display(imageNoise);
		if (draw_dispR.is_resized()) draw_dispR.resize().display(imageResult);
	}

    return 0;
}*/
