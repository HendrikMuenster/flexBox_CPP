#include <vector>
#include <iostream>

#include "CImg.h"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexLinearOperator.h"
#include "flexIdentityOperator.h"
#include "flexZeroOperator.h"
#include "flexGradientOperator.h"

#include "flexProxDualLabeling.h"
#include "flexProxDualL1Iso.h"

#include "sampleFiles.h"

using namespace std;
using namespace cimg_library;

typedef float floatingType;


int main()
{
	float weightLabelingTerm = 1.0f;
	float weightRegularizer = 0.05f;

	flexBox<floatingType> mainObject;
	mainObject.verbose = 1;

	//read original image
	CImg<unsigned char> imageOriginal(sampleFiles.imgHendrik);
	imageOriginal.resize(240, 240);

	CImg<unsigned char>	imageGray(imageOriginal.width(), imageOriginal.height(), 1, 1, 0);
	//convert to gray using luminosity method 
	for (int i = 0; i < imageOriginal.height(); ++i)
	{
		for (int j = 0; j < imageOriginal.width(); ++j)
		{
			unsigned char R = imageOriginal(i, j, 0, 0);
			unsigned char G = imageOriginal(i, j, 0, 1);
			unsigned char B = imageOriginal(i, j, 0, 2);

			unsigned char grayValue = static_cast<unsigned char>(0.299*R + 0.587*G + 0.114*B);

			imageGray(i,j) = grayValue;
		}
	}

	int numberOfLabels = 5;
	std::vector<float> labels { 0.1f, 0.3f, 0.5f, 0.7f, 0.9f };


	//add one primal variable for every label
	std::vector<int> dims;
	dims.push_back(imageGray.height());
	dims.push_back(imageGray.width());
	
	for(int i = 0; i < numberOfLabels; i++)
		mainObject.addPrimalVar(dims);

	//init data term
	//operator is a diagonal block matrix with identity operators
	int nPx = imageGray.height() * imageGray.width();
	std::vector<flexLinearOperator<floatingType>*> operatorList;
	for (int i = 0; i < numberOfLabels; i++)
	{
		for (int j = 0; j < numberOfLabels; j++)
		{
			if (i == j)
				operatorList.push_back(new flexIdentityOperator<floatingType>(nPx, nPx, false));
			else
				operatorList.push_back(new flexZeroOperator<floatingType>(nPx, nPx, false));
		}
	}

	//reshape image to vector and normalize to [0, 1]
	std::vector<floatingType> imageGrayVec(nPx, 0.0f);
	for (int i = 0; i < imageGray.height(); ++i)
	{
		for (int j = 0; j < imageGray.width(); ++j)
		{
			imageGrayVec[i*imageGray.width() + j] = static_cast<floatingType>(imageGray[i*imageGray.width() + j]) / static_cast<floatingType>(255);
		}
	}

	//calculate data part
	std::vector<std::vector<floatingType>> fList = std::vector<std::vector<floatingType>>();
	for (int i = 0; i < numberOfLabels; i++)
	{
		//push back imageGrayVec elemntwise minus label(i) and then squared
		std::vector<floatingType> tempOut(nPx, 0.0f);
		std::transform(imageGrayVec.begin(), imageGrayVec.end(), tempOut.begin(), 
			[labels, i](float elem) { float temp = elem - labels[i]; return temp * temp; });

		fList.push_back(tempOut);
	}
	

	std::vector<int> correspondingPrimals(numberOfLabels);
	std::iota(correspondingPrimals.begin(), correspondingPrimals.end(), 0); //fill correspondingPrimals with 1:numberOfLabels

	//add labeling term
	flexProx<floatingType>* labelProx = new flexProxDualLabeling<floatingType>();
	mainObject.addTerm(new flexTerm<floatingType>(labelProx, weightLabelingTerm, static_cast<int>(correspondingPrimals.size()), operatorList, fList), correspondingPrimals);
	

	for (int i = 0; i < numberOfLabels; i++)
	{
		//clear opList for next term
		operatorList.clear();


		//add regularizer termn for every label
		flexProx<floatingType>* l1IsoProx = new flexProxDualL1Iso<floatingType>();

		//add gradient for x and y direction as operators
		operatorList.push_back(new flexGradientOperator<floatingType>(mainObject.getDims(0), 0, gradientType::forward, false));
		operatorList.push_back(new flexGradientOperator<floatingType>(mainObject.getDims(0), 1, gradientType::forward, false));


		mainObject.addTerm(new flexTerm<floatingType>(l1IsoProx, weightRegularizer, 1, operatorList), { i });
	}


	mainObject.runAlgorithm();

	std::vector<float> flexResult(nPx, 0.0f); // = mainObject.getPrimal(0);

	for (int i = 0; i < numberOfLabels; i++)
	{
		std::vector<float> currentPrimalRes = mainObject.getPrimal(i);
		//multiply result with label weight (elemntwise)
		std::transform(currentPrimalRes.begin(), currentPrimalRes.end(), currentPrimalRes.begin(), std::bind1st(std::multiplies<float>(), labels[i]));

		std::transform(flexResult.begin(), flexResult.end(), currentPrimalRes.begin(), flexResult.begin(), std::plus<float>());
	}

	CImg<unsigned char>	imageResult(imageGray.width(), imageGray.height(), 1, 1, 0);
	for (int i = 0; i < imageGray.height(); ++i)
	{
		for (int j = 0; j < imageGray.width(); ++j)
		{
			imageResult(i, j, 0, 0) = static_cast<unsigned char>(flexResult[j*imageGray.width() + i] * 255);
		}
	}

	CImgDisplay main_disp(imageGray, "Original"), draw_dispR(imageResult, "Result");
	while (!main_disp.is_closed() && !main_disp.is_keyESC() && !main_disp.is_keyQ() && !draw_dispR.is_closed() && !draw_dispR.is_keyESC() && !draw_dispR.is_keyQ())
	{
		// Handle display window resizing (if any)
		if (main_disp.is_resized()) main_disp.resize().display(imageGray);
		if (draw_dispR.is_resized()) draw_dispR.resize().display(imageResult);
	}


    return 0;
}
