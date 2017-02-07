#include <vector>
#include <iostream>

#include "CImg.h"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexTermPrimal.h"
#include "flexLinearOperator.h"
#include "flexIdentityOperator.h"
#include "flexGradientOperator.h"
#include "flexProxDualDataL2.h"
#include "flexProxDualL1Iso.h"

#include "sampleFiles.h"

using namespace std;
using namespace cimg_library;

typedef float floatingType;


int main()
{
	float weightDataTerm = 1.0f;
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

	mainObject.addPrimal(new flexTermPrimal<floatingType>(1, 1, primalEmptyProx), correspondingPrimals);

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
	mainObject.addDual(new flexTermDual<floatingType>(myProx, weightDataTerm, static_cast<int>(correspondingPrimals.size()), operatorList, fList), correspondingPrimals);

	//add dualized regularizer
	operatorList.clear();
	//add gradient for x and y direction as operators
	operatorList.push_back(new flexGradientOperator<floatingType>(mainObject.getDims(0), 0, gradientType::forward, false));
	operatorList.push_back(new flexGradientOperator<floatingType>(mainObject.getDims(0), 1, gradientType::forward, false));

	flexProx<floatingType>* myProx2 = new flexProxDualL1Iso<floatingType>();
	mainObject.addDual(new flexTermDual<floatingType>(myProx2, weightRegularizer, 1, operatorList), correspondingPrimals);

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
}
