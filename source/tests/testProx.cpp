#include <vector>
#include <iostream>

//catch
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexBoxData.h"

#ifdef __CUDACC__
#include "flexBoxDataGPU.h"
#include "flexSolverPrimalDualCuda.h"
#else
#include "flexBoxDataCPU.h"
#include "flexSolverPrimalDual.h"
#endif

//proxs
#include "flexProx.h"
#include "flexProxDualBoxConstraint.h"
#include "flexProxDualDataHuber.h"
#include "flexProxDualHuber.h"
#include "flexProxDualDataKL.h"
#include "flexProxDualDataL1.h"
#include "flexProxDualDataL2.h"
#include "flexProxDualFrobenius.h"
#include "flexProxDualL1Iso.h"
#include "flexProxDualL1Aniso.h"


using namespace std;

typedef double floatingType;

#ifdef __CUDACC__
typedef thrust::device_vector<floatingType> Tdata;
#else
typedef std::vector<floatingType> Tdata;
#endif

flexBoxData<floatingType>* fData;
//flexSolver<floatingType>* fSolver;
int numElem = 64; //8x8
float weight = 0.1f;
floatingType tol = 1e-4;

void init()
{
#ifdef __CUDACC__
    fData = new flexBoxDataGPU<floatingType>();
    //fSolver = new flexSolverPrimalDualCuda<floatingType>();
    fData->
#else
    fData = new flexBoxDataCPU<floatingType>();
    //fSolver = new flexSolverPrimalDual<floatingType>();
#endif
    //fData->addPrimalVar(numElemPrimal);

#ifdef __CUDACC__
    //fData->tauElt[0] = std::vector<floatingType>(numElemPrimal, 1.0f);
    //fData->sigmaElt[0] = std::vector<floatingType>(numElemPrimal, 1.0f)
#else
    fData->tauElt.push_back(std::vector<floatingType>(numElem, 1.0f));
    fData->sigmaElt.push_back(std::vector<floatingType>(numElem, 1.0f));
    std::vector<floatingType> sequence(numElem);
    std::iota(std::begin(sequence), std::end(sequence), 1.0f);
    fData->yTilde.push_back(sequence);
    std::vector<floatingType> zeroes(numElem, 0.0f);
    fData->y.push_back(zeroes);
#endif

}

void cleanup()
{
    delete fData;
}


TEST_CASE("Prox: flexProxDualBoxConstraint<floatingType>", "[flexProxDualBoxConstraint]")
{
    init();
    auto prox = new flexProxDualBoxConstraint<floatingType>(2.0f, 6.0f);
    prox->applyProx(weight, fData, { 0 }, {}); //only corresponding first and only dual variable, primal variables are not needed

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { -1,0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i <  resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualDataHuber<floatingType>", "[flexProxDualDataHuber]")
{
    init();
    auto prox = new flexProxDualDataHuber<floatingType>(50.0f);
    std::vector<floatingType> f(numElem, 10.0f);
    std::vector<Tdata> fList;
    fList.push_back(f);
    prox->applyProx(weight, fData, { 0 }, {}, fList); //only corresponding first and only dual variable, primal variables are not needed

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { -0.0179641,-0.0159681,-0.0139721,-0.011976,-0.00998004,-0.00798403,-0.00598802,-0.00399202,-0.00199601,0,0.00199601,0.00399202,0.00598802,0.00798403,0.00998004,0.011976,0.0139721,0.0159681,0.0179641,0.0199601,0.0219561,0.0239521,0.0259481,0.0279441,0.0299401,0.0319361,0.0339321,0.0359281,0.0379242,0.0399202,0.0419162,0.0439122,0.0459082,0.0479042,0.0499002,0.0518962,0.0538922,0.0558882,0.0578842,0.0598802,0.0618762,0.0638723,0.0658683,0.0678643,0.0698603,0.0718563,0.0738523,0.0758483,0.0778443,0.0798403,0.0818363,0.0838323,0.0858283,0.0878244,0.0898204,0.0918164,0.0938124,0.0958084,0.0978044,0.0998004,0.1,0.1,0.1,0.1 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualHuber<floatingType>", "[flexProxDualHuber]")
{
    init();
    auto prox = new flexProxDualHuber<floatingType>(50.0f);
    prox->applyProx(weight, fData, { 0 }, {}); //only corresponding first and only dual variable, primal variables are not needed

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { 0.00199601,0.00399202,0.00598802,0.00798403,0.00998004,0.011976,0.0139721,0.0159681,0.0179641,0.0199601,0.0219561,0.0239521,0.0259481,0.0279441,0.0299401,0.0319361,0.0339321,0.0359281,0.0379242,0.0399202,0.0419162,0.0439122,0.0459082,0.0479042,0.0499002,0.0518962,0.0538922,0.0558882,0.0578842,0.0598802,0.0618762,0.0638723,0.0658683,0.0678643,0.0698603,0.0718563,0.0738523,0.0758483,0.0778443,0.0798403,0.0818363,0.0838323,0.0858283,0.0878244,0.0898204,0.0918164,0.0938124,0.0958084,0.0978044,0.0998004,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualDataKL<floatingType>", "[flexProxDualDataKL]")
{
    init();
    auto prox = new flexProxDualDataKL<floatingType>();
    std::vector<floatingType> f(numElem, 4000.0f);
    std::vector<Tdata> fList;
    fList.push_back(f);
    prox->applyProx(weight, fData, { 0 }, {}, fList);

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { -19.4551,-18.9725,-18.5025,-18.0448,-17.5995,-17.1664,-16.7454,-16.3363,-15.9391,-15.5535,-15.1793,-14.8163,-14.4643,-14.1232,-13.7925,-13.4721,-13.1618,-12.8612,-12.5702,-12.2884,-12.0155,-11.7514,-11.4957,-11.2481,-11.0085,-10.7765,-10.5519,-10.3345,-10.1239,-9.92003,-9.72256,-9.53129,-9.34599,-9.16645,-8.99247,-8.82383,-8.66034,-8.50181,-8.34807,-8.19894,-8.05424,-7.91381,-7.7775,-7.64516,-7.51663,-7.39179,-7.27049,-7.1526,-7.03801,-6.92659,-6.81823,-6.71282,-6.61025,-6.51043,-6.41325,-6.31863,-6.22646,-6.13668,-6.04919,-5.96392,-5.88079,-5.79973,-5.72067,-5.64353 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualDataL1<floatingType>", "[flexProxDualDataL1]")
{
    init();
    auto prox = new flexProxDualDataL1<floatingType>();
    std::vector<floatingType> f(numElem, 3.2f);
    std::vector<Tdata> fList;
    fList.push_back(f);
    prox->applyProx(10, fData, { 0 }, {}, fList);

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { -10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualDataL2<floatingType>", "[flexProxDualDataL1]")
{
    init();
    auto prox = new flexProxDualDataL2<floatingType>();
    std::vector<floatingType> f(numElem, 10.0f);
    std::vector<Tdata> fList;
    fList.push_back(f);
    prox->applyProx(weight, fData, { 0 }, {}, fList);

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { -0.818182,-0.727273,-0.636364,-0.545455,-0.454545,-0.363636,-0.272727,-0.181818,-0.0909091,0,0.0909091,0.181818,0.272727,0.363636,0.454545,0.545455,0.636364,0.727273,0.818182,0.909091,1,1.09091,1.18182,1.27273,1.36364,1.45455,1.54545,1.63636,1.72727,1.81818,1.90909,2,2.09091,2.18182,2.27273,2.36364,2.45455,2.54545,2.63636,2.72727,2.81818,2.90909,3,3.09091,3.18182,3.27273,3.36364,3.45455,3.54545,3.63636,3.72727,3.81818,3.90909,4,4.09091,4.18182,4.27273,4.36364,4.45455,4.54545,4.63636,4.72727,4.81818,4.90909 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualFrobenius<floatingType>", "[flexProxDualFrobenius]")
{
    init();
    auto prox = new flexProxDualFrobenius<floatingType>();
    prox->applyProx(10.0f, fData, { 0 }, {});

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { 0.0334375,0.066875,0.100313,0.13375,0.167188,0.200625,0.234063,0.2675,0.300938,0.334375,0.367813,0.40125,0.434688,0.468125,0.501563,0.535,0.568438,0.601875,0.635313,0.66875,0.702188,0.735626,0.769063,0.802501,0.835938,0.869376,0.902813,0.936251,0.969688,1.00313,1.03656,1.07,1.10344,1.13688,1.17031,1.20375,1.23719,1.27063,1.30406,1.3375,1.37094,1.40438,1.43781,1.47125,1.50469,1.53813,1.57156,1.605,1.63844,1.67188,1.70531,1.73875,1.77219,1.80563,1.83906,1.8725,1.90594,1.93938,1.97281,2.00625,2.03969,2.07313,2.10656,2.14 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

/*TEST_CASE("Prox: flexProxDualL1Iso<floatingType>", "[flexProxDualL1Iso]")
{
    init();
    auto prox = new flexProxDualL1Iso<floatingType>();
    prox->applyProx(32.0f, fData, { 0 }, {});

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualL1Aniso<floatingType>", "[flexProxDualL1Iso]")
{
    init();
    auto prox = new flexProxDualL1Aniso<floatingType>();
    prox->applyProx(15.0f, fData, { 0 }, {});

    SECTION("prox result")
    {
        std::vector<floatingType> resultYShould = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}*/


