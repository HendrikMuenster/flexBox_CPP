#include <vector>
#include <iostream>

//catch
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexLinearOperator.h"
#include "flexConcatOperator.h"
#include "flexDiagonalOperator.h"
#include "flexFullMatrix.h"
#include "flexGradientOperator.h"
#include "flexIdentityOperator.h"
#include "flexMatrix.h"
#include "flexMatrixLogical.h"
#include "flexSuperpixelOperator.h"
#include "flexZeroOperator.h"

using namespace std;

typedef double floatingType;

//TODO:
/*TEST_CASE("Operator flexConcatOperator<floatingType>", "[flexConcatOperator]")
{

}*/

TEST_CASE("Operator flexDiagonalOperator<floatingType>", "[flexDiagonalOperator]")
{
	floatingType tol = 1e-7;

	//add elements to diagonal operator
	std::vector<floatingType> diagonalElements = { 1.1, -2.2, 3.3, -4.4, 5.5 };

	flexDiagonalOperator<floatingType> D(diagonalElements, false);
	flexDiagonalOperator<floatingType> DM(diagonalElements, true);

	REQUIRE(D.getNumRows() == 5);
	REQUIRE(D.getNumCols() == 5);
	REQUIRE(D.isMinus == false);

	REQUIRE(DM.getNumRows() == 5);
	REQUIRE(DM.getNumCols() == 5);
	REQUIRE(DM.isMinus == true);

	SECTION("calculate row and col sums")
	{
		std::vector<floatingType> rowsSums = D.getAbsRowSum(false);
		std::vector<floatingType> colsSums = D.getAbsRowSum(true);

		std::vector<floatingType> rowsSumsM = DM.getAbsRowSum(false);
		std::vector<floatingType> colsSumsM = DM.getAbsRowSum(true);

		std::vector<floatingType> rowSumsExpected = { 1.1, 2.2, 3.3, 4.4, 5.5 };

		for (int i = 0; i < rowSumsExpected.size(); ++i)
		{
			REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);
			REQUIRE(std::abs(rowSumsExpected[i] - colsSums[i]) < tol);

			REQUIRE(std::abs(rowSumsExpected[i] - rowsSumsM[i]) < tol);
			REQUIRE(std::abs(rowSumsExpected[i] - colsSumsM[i]) < tol);
		}

	}

	SECTION("check matrix vector multiplication")
	{
		//define vector v1 = [1.1, 0.9, 4.6, -2.1, 5.9];
		std::vector<floatingType> v1 = { 1.1, 0.9, 4.6, -2.1, 5.9 };

		std::vector<floatingType> resultDv1(5, (floatingType)0);
		std::vector<floatingType> resultDtv1(5, (floatingType)0);

		std::vector<floatingType> resultDMv1(5, (floatingType)0);
		std::vector<floatingType> resultDMtv1(5, (floatingType)0);

		D.timesPlus(false, v1, resultDv1);
		D.timesPlus(true, v1, resultDtv1);

		DM.timesPlus(false, v1, resultDMv1);
		DM.timesPlus(true, v1, resultDMtv1);

		std::vector<floatingType> resultDv1Expected = { 1.21, -1.98, 15.18, 9.24, 32.45 };
		std::vector<floatingType> resultDMv1Expected = { -1.21, 1.98, -15.18, -9.24, -32.45 };

		for (int i = 0; i < resultDv1Expected.size(); ++i)
		{
			REQUIRE(std::abs(resultDv1[i] - resultDv1Expected[i]) < tol);
			REQUIRE(std::abs(resultDtv1[i] - resultDv1Expected[i]) < tol);

			REQUIRE(std::abs(resultDMv1[i] - resultDMv1Expected[i]) < tol);
			REQUIRE(std::abs(resultDMtv1[i] - resultDMv1Expected[i]) < tol);
		}
	}

	SECTION("check empty matrix")
	{
		std::vector<floatingType> emptyValVec = {};
		flexDiagonalOperator<floatingType> empty(emptyValVec, false);


		REQUIRE(empty.getNumRows() == 0);
		REQUIRE(empty.getNumCols() == 0);
		REQUIRE(empty.isMinus == false);

		std::vector<floatingType> v = {};
		std::vector<floatingType> resultAv = {};

		empty.timesPlus(false, v, resultAv);
		REQUIRE(resultAv.size() == 0);
	}
}

TEST_CASE("Operator flexFullMatrix<floatingType>", "[flexFullMatrix]")
{
	floatingType tol = 1e-7;
	flexFullMatrix<floatingType> A(5, 3, false);

	//add elements to matrix. Structure is:
	//A=	[5.0	1.0		6.3;
	//		1.21	65.1	-0.1;
	//		3.2		-12.0	0.01;
	//		-10.0	1.0		5.0;
	//		-4.0	3.0		1.115]
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
	//std::vector<floatingType> v = { (floatingType)-1.012, (floatingType)2.7, (floatingType)500 };

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

	SECTION("check empty matrix")
	{
		flexFullMatrix<floatingType> empty(0, 0, false);

		REQUIRE(empty.getNumRows() == 0);
		REQUIRE(empty.getNumCols() == 0);
		REQUIRE(empty.isMinus == false);

		std::vector<floatingType> v = {};
		std::vector<floatingType> resultAv = {};

		empty.timesPlus(false, v, resultAv);
		REQUIRE(resultAv.size() == 0);
	}
}


TEST_CASE("Operator flexGradientOperator<floatingType>", "[flexGradientOperator]")
{
	floatingType tol = 1e-7;
	flexGradientOperator<floatingType> gradOpForward({ 3, 4 }, 0, gradientType::forward, false);
	flexGradientOperator<floatingType> gradOpBackward({ 3, 4 }, 0, gradientType::backward, false);

	REQUIRE(gradOpForward.getNumRows() == 12);
	REQUIRE(gradOpForward.getNumCols() == 12);
	REQUIRE(gradOpForward.isMinus == false);

	REQUIRE(gradOpBackward.getNumRows() == 12);
	REQUIRE(gradOpBackward.getNumCols() == 12);
	REQUIRE(gradOpBackward.isMinus == false);


	SECTION("check matrix vector multiplication")
	{
		//define vector v = [1,2,3....];
		std::vector<floatingType> v(12); //3*4
		std::iota(v.begin(), v.end(), 1);

		std::vector<floatingType> resultForward(12, (floatingType)0);
		std::vector<floatingType> resultBackward(12, (floatingType)0);

		gradOpForward.times(false, v, resultForward);
		gradOpBackward.times(false, v, resultBackward);

		std::vector<floatingType> resultForwardExpected = { 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0};
		std::vector<floatingType> resultBackwardExpected = { 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1};

		for (int i = 0; i < resultForward.size(); ++i)
		{
			REQUIRE(std::abs(resultForward[i] - resultForwardExpected[i]) < tol);
			//REQUIRE(std::abs(resultBackward[i] - resultBackwardExpected[i]) < tol);
		}
	}



}

TEST_CASE("Operator flexIdentityOperator<floatingType>", "[flexIdentityOperator]")
{
	floatingType tol = 1e-7;
	flexIdentityOperator<floatingType> ID(3, 5, false);

	REQUIRE(ID.getNumRows() == 3);
	REQUIRE(ID.getNumCols() == 5);
	REQUIRE(ID.isMinus == false);

	SECTION("calculate row and col sums")
	{
		std::vector<floatingType> rowsSums = ID.getAbsRowSum(false);
		std::vector<floatingType> colsSums = ID.getAbsRowSum(true);

		std::vector<floatingType> rowSumsExpected(3, 1.0);
		std::vector<floatingType> colSumsExpected(5, 1.0);


		for (int i = 0; i < rowSumsExpected.size(); ++i)
			REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

		for (int i = 0; i < colSumsExpected.size(); ++i)
			REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
	}

	SECTION("check matrix vector multiplication")
	{
		//define vector v1 = [1.1, 0.9, 4.6];
		std::vector<floatingType> v1 = { 1.1, 0.9, 4.6, -3.5, -100.2};
		//define vector v2 = [1.1, 0.9];
		std::vector<floatingType> v2 = { 1.1, 0.9, 4.6};

		std::vector<floatingType> resultIDv1(3, (floatingType)0);
		std::vector<floatingType> resultIDtv2(5, (floatingType)0);

		ID.times(false, v1, resultIDv1);
		ID.times(true, v2, resultIDtv2);

		std::vector<floatingType> resultIDv1Expected = { 1.1, 0.9, 4.6 };
		std::vector<floatingType> resultIDtv2Expected = { 1.1, 0.9, 4.6, 0, 0 };

		for (int i = 0; i < resultIDv1.size(); ++i)
			REQUIRE(std::abs(resultIDv1[i] - resultIDv1Expected[i]) < tol);
		for (int i = 0; i < resultIDtv2.size(); ++i)
			REQUIRE(std::abs(resultIDtv2[i] - resultIDtv2Expected[i]) < tol);
	}

	SECTION("check empty matrix")
	{
		flexIdentityOperator<floatingType> empty(0, 0, false);

		REQUIRE(empty.getNumRows() == 0);
		REQUIRE(empty.getNumCols() == 0);
		REQUIRE(empty.isMinus == false);

		std::vector<floatingType> v = {};
		std::vector<floatingType> resultIDv = {};

		empty.timesPlus(false, v, resultIDv);
		REQUIRE(resultIDv.size() == 0);
	}
}

TEST_CASE("Operator flexMatrix<floatingType>", "[flexMatrix]")
{
	floatingType tol = 1e-7;
	flexMatrix<floatingType> A(2, 3, false);

	//add elements to matrix. Structure is:
	//A =	[3.1 2.2 1.5;
	//		 1.2 0.9 2.4 ]
	std::vector<int> rows = { 0, 0, 0, 1, 1, 1 };
	std::vector<int> cols = { 0, 1, 2, 0, 1, 2 };
	std::vector<floatingType> vals = { 3.1, 2.2, 1.5, 1.2, 0.9, 2.4 };
	A.blockInsert(rows, cols, vals);

	REQUIRE(A.getNumRows() == 2);
	REQUIRE(A.getNumCols() == 3);
	REQUIRE(A.isMinus == false);

	SECTION("calculate row and col sums")
	{
		std::vector<floatingType> rowsSums = A.getAbsRowSum(false);
		std::vector<floatingType> colsSums = A.getAbsRowSum(true);

		std::vector<floatingType> rowSumsExpected = { 6.8, 4.5};
		std::vector<floatingType> colSumsExpected = { 4.3, 3.1, 3.9};

		for (int i = 0; i < rowSumsExpected.size(); ++i)
			REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

		for (int i = 0; i < colSumsExpected.size(); ++i)
			REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
	}

	SECTION("check matrix vector multiplication")
	{
		//define vector v1 = [1.1, 0.9, 4.6];
		std::vector<floatingType> v1 = { 1.1, 0.9, 4.6 };
		//define vector v2 = [1.1, 0.9];
		std::vector<floatingType> v2 = { 1.1, 0.9 };

		std::vector<floatingType> resultAv1(2, (floatingType)0);
		std::vector<floatingType> resultAtv2(3, (floatingType)0);

		A.timesPlus(false, v1, resultAv1);
		A.timesPlus(true, v2, resultAtv2);

		std::vector<floatingType> resultAv1Expected = { 12.29, 13.17 };
		std::vector<floatingType> resultAtv2Expected = { 4.49, 3.23, 3.81 };

		for (int i = 0; i < resultAv1Expected.size(); ++i)
			REQUIRE(std::abs(resultAv1[i] - resultAv1Expected[i]) < tol);
		for (int i = 0; i < resultAtv2Expected.size(); ++i)
			REQUIRE(std::abs(resultAtv2[i] - resultAtv2Expected[i]) < tol);
	}

	SECTION("check empty matrix")
	{
		flexMatrix<floatingType> empty(0, 0, false);
		std::vector<int> emptyIntVec = {};
		std::vector<floatingType> emptyFloatVec = {};
		empty.blockInsert(emptyIntVec, emptyIntVec, emptyFloatVec);

		REQUIRE(empty.getNumRows() == 0);
		REQUIRE(empty.getNumCols() == 0);
		REQUIRE(empty.isMinus == false);

		std::vector<floatingType> v = {};
		std::vector<floatingType> resultAv = {};

		empty.timesPlus(false, v, resultAv);
		REQUIRE(resultAv.size() == 0);
	}
}

TEST_CASE("Operator flexMatrixLogical<floatingType>", "[flexMatrixLogical]")
{
	floatingType tol = 1e-7;
	flexMatrixLogical<floatingType> A(2, 3, false);

	//add elements to matrix. Structure is:
	//A =	[1 1 0;
	//		 0 1 1 ]
	std::vector<int> rows = { 0, 0, 1, 1 };
	std::vector<int> cols = { 0, 1, 1, 2 };
	A.blockInsert(rows, cols);

	REQUIRE(A.getNumRows() == 2);
	REQUIRE(A.getNumCols() == 3);
	REQUIRE(A.isMinus == false);

	SECTION("calculate row and col sums")
	{
		std::vector<floatingType> rowsSums = A.getAbsRowSum(false);
		std::vector<floatingType> colsSums = A.getAbsRowSum(true);

		std::vector<floatingType> rowSumsExpected = { 2, 2 };
		std::vector<floatingType> colSumsExpected = { 1, 2, 1 };

		for (int i = 0; i < rowSumsExpected.size(); ++i)
			REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

		for (int i = 0; i < colSumsExpected.size(); ++i)
			REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
	}

	SECTION("check matrix vector multiplication")
	{
		//define vector v1 = [1.1, 0.9, 4.6];
		std::vector<floatingType> v1 = { 1.1, 0.9, 4.6 };
		//define vector v2 = [1.1, 0.9];
		std::vector<floatingType> v2 = { 1.1, 0.9 };

		std::vector<floatingType> resultAv1(2, (floatingType)0);
		std::vector<floatingType> resultAtv2(3, (floatingType)0);

		A.timesPlus(false, v1, resultAv1);
		A.timesPlus(true, v2, resultAtv2);

		std::vector<floatingType> resultAv1Expected = { 2, 5.5 };
		std::vector<floatingType> resultAtv2Expected = { 1.1, 2, 0.9 };

		for (int i = 0; i < resultAv1Expected.size(); ++i)
			REQUIRE(std::abs(resultAv1[i] - resultAv1Expected[i]) < tol);
		for (int i = 0; i < resultAtv2Expected.size(); ++i)
			REQUIRE(std::abs(resultAtv2[i] - resultAtv2Expected[i]) < tol);
	}

	SECTION("check empty matrix")
	{
		flexMatrixLogical<floatingType> empty(0, 0, false);
		std::vector<int> emptyIntVec = {};
		std::vector<floatingType> emptyFloatVec = {};
		empty.blockInsert(emptyIntVec, emptyIntVec);

		REQUIRE(empty.getNumRows() == 0);
		REQUIRE(empty.getNumCols() == 0);
		REQUIRE(empty.isMinus == false);

		std::vector<floatingType> v = {};
		std::vector<floatingType> resultAv = {};

		empty.timesPlus(false, v, resultAv);
		REQUIRE(resultAv.size() == 0);
	}

}

TEST_CASE("Operator flexSuperpixelOperator<floatingType>", "[flexSuperpixelOperator]")
{
	floatingType tol = 1e-7;
	//target dimension is [3,2] with upsampling factor of 2 (so [6,4])
	flexSuperpixelOperator<floatingType> A({ 3, 2 }, 2, false);

	REQUIRE(A.getNumRows() == 6); //prod of all dimensions on target dim == 3*2
	REQUIRE(A.getNumCols() == 24); //prod of all dimensions on target dim times upsampling factor squared == (3*2)*(2*2)
	REQUIRE(A.isMinus == false);

	SECTION("calculate row and col sums")
	{
		std::vector<floatingType> rowsSums = A.getAbsRowSum(false);
		std::vector<floatingType> colsSums = A.getAbsRowSum(true);

		std::vector<floatingType> rowSumsExpected(6);
		std::vector<floatingType> colSumsExpected(24);
		std::fill(rowSumsExpected.begin(), rowSumsExpected.end(), 1);
		std::fill(colSumsExpected.begin(), colSumsExpected.end(), 0.25); //== 1/(upsam Factor)^2

		for (int i = 0; i < rowSumsExpected.size(); ++i)
			REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

		for (int i = 0; i < colSumsExpected.size(); ++i)
			REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
	}

	SECTION("check matrix vector multiplication")
	{
		//define vector v1 = [1.1, 0.9, 4.6];
		std::vector<floatingType> v1 = { 8, -9, 9, -2, -10, -3, 5, 6, 1, 4, 8, -9, -4, -10, -6, 5, 5, 8, 2, -9, 9, 6, -4, 1 };

		std::vector<floatingType> resultAv1(6, (floatingType)0);

		A.timesPlus(false, v1, resultAv1);

		std::vector<floatingType> resultAv1Expected = { -3.5, 4.5, -2.25, -0.5, 7, -2.5};

		for (int i = 0; i < resultAv1Expected.size(); ++i)
			REQUIRE(std::abs(resultAv1[i] - resultAv1Expected[i]) < tol);
	}

}

TEST_CASE("Operator flexZeroOperator<floatingType>", "[flexZeroOperator]")
{
	floatingType tol = 1e-7;
	flexZeroOperator<floatingType> A(2, 3, false);

	REQUIRE(A.getNumRows() == 2);
	REQUIRE(A.getNumCols() == 3);
	REQUIRE(A.isMinus == false);

	SECTION("calculate row and col sums")
	{
		std::vector<floatingType> rowsSums = A.getAbsRowSum(false);
		std::vector<floatingType> colsSums = A.getAbsRowSum(true);

		std::vector<floatingType> rowSumsExpected = { 0, 0 };
		std::vector<floatingType> colSumsExpected = { 0, 0, 0};

		for (int i = 0; i < rowSumsExpected.size(); ++i)
			REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

		for (int i = 0; i < colSumsExpected.size(); ++i)
			REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
	}

	SECTION("check matrix vector multiplication")
	{
		//define vector v1 = [1.1, 0.9, 4.6];
		std::vector<floatingType> v1 = { 1.1, 0.9, 4.6 };
		//define vector v2 = [1.1, 0.9];
		std::vector<floatingType> v2 = { 1.1, 0.9 };

		std::vector<floatingType> resultAv1(2, (floatingType)0);
		std::vector<floatingType> resultAtv2(3, (floatingType)0);

		A.times(false, v1, resultAv1);
		A.times(true, v2, resultAtv2);

		std::vector<floatingType> resultAv1Expected = { 0, 0 };
		std::vector<floatingType> resultAtv2Expected = { 0, 0, 0 };

		for (int i = 0; i < resultAv1Expected.size(); ++i)
			REQUIRE(std::abs(resultAv1[i] - resultAv1Expected[i]) < tol);
		for (int i = 0; i < resultAtv2Expected.size(); ++i)
			REQUIRE(std::abs(resultAtv2[i] - resultAtv2Expected[i]) < tol);
	}

	SECTION("check empty matrix")
	{
		flexZeroOperator<floatingType> empty(0, 0, false);
		std::vector<int> emptyIntVec = {};
		std::vector<floatingType> emptyFloatVec = {};

		REQUIRE(empty.getNumRows() == 0);
		REQUIRE(empty.getNumCols() == 0);
		REQUIRE(empty.isMinus == false);

		std::vector<floatingType> v = {};
		std::vector<floatingType> resultAv = {};

		empty.times(false, v, resultAv);
		REQUIRE(resultAv.size() == 0);
	}

}
