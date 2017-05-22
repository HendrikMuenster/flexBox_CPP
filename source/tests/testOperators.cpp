#include <vector>
#include <iostream>

//catch
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexLinearOperator.h"
#include "flexFullMatrix.h"
#include "flexMatrix.h"
#include "flexIdentityOperator.h"
#include "flexGradientOperator.h"
#include "flexProxDualDataL2.h"
#include "flexProxDualL1Iso.h"

using namespace std;

typedef double floatingType;


TEST_CASE("Operator flexFullMatrix<floatingType>", "[flexFullMatrix]")
{
	floatingType tol = 1e-7;
	flexFullMatrix<floatingType> A(5, 3, false);

	//add elements to matrix. Structure is:
	/*A=	[5.0	1.0		6.3;
			1.21	65.1	-0.1;
			3.2		-12.0	0.01;
			-10.0	1.0		5.0;
			-4.0	3.0		1.115]*/
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

TEST_CASE("Operator flexMatrix<floatingType>", "[flexMatrix]")
{
	floatingType tol = 1e-7;
	flexMatrix<floatingType> A(2, 3, false);

	//add elements to matrix. Structure is:
	/*A =	[3.1 2.2 1.5;
			 1.2 0.9 2.4 ]*/
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
