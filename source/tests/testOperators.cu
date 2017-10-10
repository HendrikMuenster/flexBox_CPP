#include <vector>
#include <iostream>

#include <thrust/device_vector.h>

//catch
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexLinearOperator.h"
#include "flexConcatOperator.h"
#include "flexDiagonalOperator.h"
#include "flexGradientOperator.h"
#include "flexIdentityOperator.h"
#include "flexMatrixGPU.h"
#include "flexMatrixLogical.h"
#include "flexSuperpixelOperator.h"
#include "flexZeroOperator.h"

using namespace std;

typedef float floatingType;

//TODO:
/*TEST_CASE("Operator flexConcatOperator<floatingType>", "[flexConcatOperator]")
{

}*/

TEST_CASE("Operator flexDiagonalOperator<floatingType>", "[flexDiagonalOperator]")
{
    floatingType tol = 1e-4;

    //add elements to diagonal operator
    std::vector<floatingType> diagonalElements = { 1.1f, -2.2f, 3.3f, -4.4f, 5.5f };

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

        std::vector<floatingType> rowSumsExpected = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f };

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
        std::vector<floatingType> v1Host = { 1.1f, 0.9f, 4.6f, -2.1f, 5.9f };
        thrust::device_vector<floatingType> v1(v1Host);

        thrust::device_vector<floatingType> resultDv1(5, (floatingType)0);
        thrust::device_vector<floatingType> resultDtv1(5, (floatingType)0);

        thrust::device_vector<floatingType> resultDMv1(5, (floatingType)0);
        thrust::device_vector<floatingType> resultDMtv1(5, (floatingType)0);

        D.timesPlus(false, v1, resultDv1);
        D.timesPlus(true, v1, resultDtv1);

        DM.timesPlus(false, v1, resultDMv1);
        DM.timesPlus(true, v1, resultDMtv1);

        std::vector<floatingType> resultDv1Expected = { 1.21f, -1.98f, 15.18f, 9.24f, 32.45f };
        std::vector<floatingType> resultDMv1Expected = { -1.21f, 1.98f, -15.18f, -9.24f, -32.45f };

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

        thrust::device_vector<floatingType> v;
        thrust::device_vector<floatingType> resultAv;

        empty.timesPlus(false, v, resultAv);
        REQUIRE(resultAv.size() == 0);
    }
}

/*TEST_CASE("Operator flexGradientOperator<floatingType>", "[flexGradientOperator]")
{
    floatingType tol = 1e-3f;
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
        thrust::device_vector<floatingType> v(12); //3*4
        thrust::sequence(v.begin(), v.end(), (floatingType)1.0);

        thrust::device_vector<floatingType> resultForward(12, (floatingType)0);
        thrust::device_vector<floatingType> resultBackward(12, (floatingType)0);

        gradOpForward.times(false, v, resultForward);
        gradOpBackward.times(false, v, resultBackward);

        std::vector<floatingType> resultForwardExpected = { 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0 };
        std::vector<floatingType> resultBackwardExpected = { 1, 1, -2, 4, 1, -5, 7, 1, -8, 10, 1, -11 };

        for (int i = 0; i < resultForward.size(); ++i)
        {
            REQUIRE(std::abs(resultForward[i] - resultForwardExpected[i]) < tol);
            REQUIRE(std::abs(resultBackward[i] - resultBackwardExpected[i]) < tol);
        }
    }
}*/

TEST_CASE("Operator flexIdentityOperator<floatingType>", "[flexIdentityOperator]")
{
    floatingType tol = 1e-4f;
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
        std::vector<floatingType> v1Host = { 1.1f, 0.9f, 4.6f, -3.5f, -100.2f };
        thrust::device_vector<floatingType> v1(v1Host);
        //define vector v2 = [1.1, 0.9];
        std::vector<floatingType> v2Host = { 1.1f, 0.9f, 4.6f };
        thrust::device_vector<floatingType> v2(v2Host);

        thrust::device_vector<floatingType> resultIDv1(3, (floatingType)0);
        thrust::device_vector<floatingType> resultIDtv2(5, (floatingType)0);

        ID.times(false, v1, resultIDv1);
        ID.times(true, v2, resultIDtv2);

        std::vector<floatingType> resultIDv1Expected = { 1.1f, 0.9f, 4.6f };
        std::vector<floatingType> resultIDtv2Expected = { 1.1f, 0.9f, 4.6f, 0.0f, 0.0f };

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

        thrust::device_vector<floatingType> v;
        thrust::device_vector<floatingType> resultIDv;

        empty.timesPlus(false, v, resultIDv);
        REQUIRE(resultIDv.size() == 0);
    }
}

/*TEST_CASE("Operator flexMatrixGPU<floatingType>", "[flexMatrixGPU]")
{
    floatingType tol = 1e-4f;

    //add elements to matrix. Structure is:
    //A =	[3.1 2.2 1.5;
    //		 1.2 0.9 2.4 ]
    std::vector<int> rows = { 0, 0, 0, 1, 1, 1 };
    std::vector<int> cols = { 0, 1, 2, 0, 1, 2 };
    std::vector<floatingType> vals = { 3.1f, 2.2f, 1.5f, 1.2f, 0.9f, 2.4f };
    flexMatrixGPU<floatingType> A(2, 3, rows.data(), cols.data(), vals.data(), true, false);
    //A.blockInsert(rows, cols, vals);

    REQUIRE(A.getNumRows() == 2);
    REQUIRE(A.getNumCols() == 3);
    REQUIRE(A.isMinus == false);

    SECTION("calculate row and col sums")
    {
        std::vector<floatingType> rowsSums = A.getAbsRowSum(false);
        std::vector<floatingType> colsSums = A.getAbsRowSum(true);

        std::vector<floatingType> rowSumsExpected = { 6.8f, 4.5f };
        std::vector<floatingType> colSumsExpected = { 4.3f, 3.1f, 3.9f };

        for (int i = 0; i < rowSumsExpected.size(); ++i)
            REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

        for (int i = 0; i < colSumsExpected.size(); ++i)
            REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
    }

    SECTION("check matrix vector multiplication")
    {
        //define vector v1 = [1.1, 0.9, 4.6];
        std::vector<floatingType> v1Host = { 1.1f, 0.9f, 4.6f };
        thrust::device_vector<floatingType> v1(v1Host);
        //define vector v2 = [1.1, 0.9];
        std::vector<floatingType> v2Host = { 1.1f, 0.9f };
        thrust::device_vector<floatingType> v2(v2Host);

        thrust::device_vector<floatingType> resultAv1(2, (floatingType)0);
        thrust::device_vector<floatingType> resultAtv2(3, (floatingType)0);

        A.timesPlus(false, v1, resultAv1);
        A.timesPlus(true, v2, resultAtv2);

        std::vector<floatingType> resultAv1Expected = { 12.29f, 13.17f };
        std::vector<floatingType> resultAtv2Expected = { 4.49f, 3.23f, 3.81f };

        for (int i = 0; i < resultAv1Expected.size(); ++i)
            REQUIRE(std::abs(resultAv1[i] - resultAv1Expected[i]) < tol);
        for (int i = 0; i < resultAtv2Expected.size(); ++i)
            REQUIRE(std::abs(resultAtv2[i] - resultAtv2Expected[i]) < tol);
    }

    SECTION("check empty matrix")
    {
        std::vector<int> emptyIntVec = {};
        std::vector<floatingType> emptyFloatVec = {};
        flexMatrixGPU<floatingType> empty(0, 0, emptyIntVec.data(), emptyIntVec.data(), emptyFloatVec.data(), true, false);

        REQUIRE(empty.getNumRows() == 0);
        REQUIRE(empty.getNumCols() == 0);
        REQUIRE(empty.isMinus == false);

        thrust::device_vector<floatingType> v;
        thrust::device_vector<floatingType> resultAv;

        empty.timesPlus(false, v, resultAv);
        REQUIRE(resultAv.size() == 0);
    }
}*/

TEST_CASE("Operator flexSuperpixelOperator<floatingType>", "[flexSuperpixelOperator]")
{
    floatingType tol = 1e-4f;
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
        std::vector<floatingType> v1Host = { 8, -9, 9, -2, -10, -3, 5, 6, 1, 4, 8, -9, -4, -10, -6, 5, 5, 8, 2, -9, 9, 6, -4, 1 };
        thrust::device_vector<floatingType> v1(v1Host);

        thrust::device_vector<floatingType> resultAv1(6, (floatingType)0);

        A.timesPlus(false, v1, resultAv1);

        std::vector<floatingType> resultAv1Expected = { -3.5, 4.5, -2.25, -0.5, 7, -2.5 };

        for (int i = 0; i < resultAv1Expected.size(); ++i)
            REQUIRE(std::abs(resultAv1[i] - resultAv1Expected[i]) < tol);
    }

}

TEST_CASE("Operator flexZeroOperator<floatingType>", "[flexZeroOperator]")
{
    floatingType tol = 1e-4;
    flexZeroOperator<floatingType> A(2, 3, false);

    REQUIRE(A.getNumRows() == 2);
    REQUIRE(A.getNumCols() == 3);
    REQUIRE(A.isMinus == false);

    SECTION("calculate row and col sums")
    {
        std::vector<floatingType> rowsSums = A.getAbsRowSum(false);
        std::vector<floatingType> colsSums = A.getAbsRowSum(true);

        std::vector<floatingType> rowSumsExpected = { 0, 0 };
        std::vector<floatingType> colSumsExpected = { 0, 0, 0 };

        for (int i = 0; i < rowSumsExpected.size(); ++i)
            REQUIRE(std::abs(rowSumsExpected[i] - rowsSums[i]) < tol);

        for (int i = 0; i < colSumsExpected.size(); ++i)
            REQUIRE(std::abs(colSumsExpected[i] - colsSums[i]) < tol);
    }

    SECTION("check matrix vector multiplication")
    {
        //define vector v1 = [1.1, 0.9, 4.6];
        std::vector<floatingType> v1Host = { 1.1f, 0.9f, 4.6f };
        thrust::device_vector<floatingType> v1(v1Host);
        //define vector v2 = [1.1, 0.9];
        std::vector<floatingType> v2Host = { 1.1f, 0.9f };
        thrust::device_vector<floatingType> v2(v2Host);

        thrust::device_vector<floatingType> resultAv1(2, (floatingType)0);
        thrust::device_vector<floatingType> resultAtv2(3, (floatingType)0);

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

        thrust::device_vector<floatingType> v;
        thrust::device_vector<floatingType> resultAv;

        empty.times(false, v, resultAv);
        REQUIRE(resultAv.size() == 0);
    }

}
