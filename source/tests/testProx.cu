//std
#include <vector>
#include <iostream>

//CUDA
#include <thrust/device_vector.h>

//catch
#include "catch.hpp"

//flexBox
#include "tools.h"
#include "flexBox.h"
#include "flexBoxData.h"
#include "flexBoxDataGPU.h"

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
#include "flexProxDualL2.h"
#include "flexProxDualL2Inf.h"
#include "flexProxDualLInf.h"

#include "flexProxDualLabeling.h"


using namespace std;

typedef float floatingType;
typedef thrust::device_vector<floatingType> Tdata;


flexBoxData<floatingType>* fData;
int numElem = 64; //64 x 2 if 2D
float weight = 0.1f;
floatingType tol = 1e-4;

void init1d()
{
    fData = new flexBoxDataGPU<floatingType>();

    fData->tauElt.push_back(thrust::device_vector<floatingType>(numElem, 1.0f));
    fData->sigmaElt.push_back(thrust::device_vector<floatingType>(numElem, 1.0f));

    thrust::device_vector<floatingType> seq(numElem);
    thrust::sequence(seq.begin(), seq.end(), 1.0f);
    fData->yTilde.push_back(seq);

    fData->y.push_back(thrust::device_vector<floatingType>(numElem, 0.0f));
}

void init2d()
{
    init1d();
    fData->sigmaElt.push_back(thrust::device_vector<floatingType>(numElem, 1.0f));

    thrust::device_vector<floatingType> seq(numElem);
    thrust::sequence(seq.begin(), seq.end(), 2.0f);
    fData->yTilde.push_back(seq);

    fData->y.push_back(thrust::device_vector<floatingType>(numElem, 0.0f));
}

void cleanup()
{
    delete fData;
}


TEST_CASE("Prox: flexProxDualBoxConstraint<floatingType>", "[flexProxDualBoxConstraint]")
{
    init1d();
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


TEST_CASE("Prox: flexProxDualDataKL<floatingType>", "[flexProxDualDataKL]")
{
    init1d();
    auto prox = new flexProxDualDataKL<floatingType>();
    std::vector<floatingType> f(numElem, 4000.0f);
    std::vector<Tdata> fList;
    fList.push_back(f);
    prox->applyProx(weight, fData, { 0 }, {}, fList);

    SECTION("prox result")
    {
        std::vector<double> tmp = { -19.4551,-18.9725,-18.5025,-18.0448,-17.5995,-17.1664,-16.7454,-16.3363,-15.9391,-15.5535,-15.1793,-14.8163,-14.4643,-14.1232,-13.7925,-13.4721,-13.1618,-12.8612,-12.5702,-12.2884,-12.0155,-11.7514,-11.4957,-11.2481,-11.0085,-10.7765,-10.5519,-10.3345,-10.1239,-9.92003,-9.72256,-9.53129,-9.34599,-9.16645,-8.99247,-8.82383,-8.66034,-8.50181,-8.34807,-8.19894,-8.05424,-7.91381,-7.7775,-7.64516,-7.51663,-7.39179,-7.27049,-7.1526,-7.03801,-6.92659,-6.81823,-6.71282,-6.61025,-6.51043,-6.41325,-6.31863,-6.22646,-6.13668,-6.04919,-5.96392,-5.88079,-5.79973,-5.72067,-5.64353 };
        std::vector<floatingType> resultYShould(std::begin(tmp), std::end(tmp));
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualDataL1<floatingType>", "[flexProxDualDataL1]")
{
    init1d();
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
    init1d();
    auto prox = new flexProxDualDataL2<floatingType>();
    std::vector<floatingType> f(numElem, 10.0f);
    std::vector<Tdata> fList;
    fList.push_back(f);
    prox->applyProx(weight, fData, { 0 }, {}, fList);

    SECTION("prox result")
    {
        std::vector<double> tmp = { -0.818182,-0.727273,-0.636364,-0.545455,-0.454545,-0.363636,-0.272727,-0.181818,-0.0909091,0,0.0909091,0.181818,0.272727,0.363636,0.454545,0.545455,0.636364,0.727273,0.818182,0.909091,1,1.09091,1.18182,1.27273,1.36364,1.45455,1.54545,1.63636,1.72727,1.81818,1.90909,2,2.09091,2.18182,2.27273,2.36364,2.45455,2.54545,2.63636,2.72727,2.81818,2.90909,3,3.09091,3.18182,3.27273,3.36364,3.45455,3.54545,3.63636,3.72727,3.81818,3.90909,4,4.09091,4.18182,4.27273,4.36364,4.45455,4.54545,4.63636,4.72727,4.81818,4.90909 };
        std::vector<floatingType> resultYShould(std::begin(tmp), std::end(tmp));
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualL2<floatingType>", "[flexProxDualL2]")
{
    init1d();
    auto prox = new flexProxDualL2<floatingType>();
    prox->applyProx(10.0f, fData, { 0 }, {});

    SECTION("prox result")
    {
        std::vector<double> tmp = { 0.909091,1.81818,2.72727,3.63636,4.54545,5.45455,6.36364,7.27273,8.18182,9.09091,10,10.9091,11.8182,12.7273,13.6364,14.5455,15.4545,16.3636,17.2727,18.1818,19.0909,20,20.9091,21.8182,22.7273,23.6364,24.5455,25.4545,26.3636,27.2727,28.1818,29.0909,30,30.9091,31.8182,32.7273,33.6364,34.5455,35.4545,36.3636,37.2727,38.1818,39.0909,40,40.9091,41.8182,42.7273,43.6364,44.5455,45.4545,46.3636,47.2727,48.1818,49.0909,50,50.9091,51.8182,52.7273,53.6364,54.5455,55.4545,56.3636,57.2727,58.1818 };
        std::vector<floatingType> resultYShould(std::begin(tmp), std::end(tmp));
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualFrobenius<floatingType>", "[flexProxDualFrobenius]")
{
    init1d();
    auto prox = new flexProxDualFrobenius<floatingType>();
    prox->applyProx(10.0f, fData, { 0 }, {});

    SECTION("prox result")
    {
        std::vector<double> tmp = { 0.0334375,0.066875,0.100313,0.13375,0.167188,0.200625,0.234063,0.2675,0.300938,0.334375,0.367813,0.40125,0.434688,0.468125,0.501563,0.535,0.568438,0.601875,0.635313,0.66875,0.702188,0.735626,0.769063,0.802501,0.835938,0.869376,0.902813,0.936251,0.969688,1.00313,1.03656,1.07,1.10344,1.13688,1.17031,1.20375,1.23719,1.27063,1.30406,1.3375,1.37094,1.40438,1.43781,1.47125,1.50469,1.53813,1.57156,1.605,1.63844,1.67188,1.70531,1.73875,1.77219,1.80563,1.83906,1.8725,1.90594,1.93938,1.97281,2.00625,2.03969,2.07313,2.10656,2.14 };
        std::vector<floatingType> resultYShould(std::begin(tmp), std::end(tmp));
        auto resultYIs = fData->y[0];
        for (size_t i = 0; i < resultYIs.size(); i++)
            REQUIRE(std::abs(resultYIs[i] - resultYShould[i]) < tol);
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualL1Iso<floatingType>", "[flexProxDualL1Iso]")
{
    init2d();
    auto prox = new flexProxDualL1Iso<floatingType>();
    prox->applyProx(1.0f, fData, { 0, 1 }, {});

    SECTION("prox result")
    {
        std::vector<double> tmp = { 0.4472136,0.5547002,0.6,0.624695,0.6401844,0.6507914,0.6585046,0.6643638,0.6689647,0.6726728,0.6757246,0.6782801,0.6804511,0.6823183,0.6839411,0.6853647,0.6866235,0.6877446,0.6887495,0.6896552,0.6904757,0.6912226,0.6919054,0.6925318,0.6931087,0.6936417,0.6941356,0.6945945,0.6950221,0.6954214,0.6957952,0.6961458,0.6964754,0.6967857,0.6970784,0.697355,0.6976167,0.6978647,0.6981001,0.6983239,0.6985367,0.6987395,0.6989329,0.6991176,0.6992942,0.699463,0.6996248,0.6997798,0.6999286,0.7000714,0.7002087,0.7003407,0.7004677,0.7005901,0.700708,0.7008218,0.7009315,0.7010375,0.70114,0.701239,0.7013348,0.7014275,0.7015173,0.7016043 };
        std::vector<floatingType> resultYShould0(std::begin(tmp), std::end(tmp));
        tmp = { 0.8944272,0.8320503,0.8,0.7808688,0.7682213,0.7592566,0.7525767,0.7474093,0.7432941,0.7399401,0.7371541,0.7348034,0.7327935,0.7310553,0.7295372,0.7282,0.7270132,0.7259527,0.7249994,0.7241379,0.7233555,0.7226419,0.7219882,0.7213873,0.7208331,0.7203202,0.7198443,0.7194015,0.7189884,0.7186021,0.7182402,0.7179004,0.7175807,0.7172794,0.7169949,0.7167259,0.7164712,0.7162296,0.7160001,0.7157819,0.7155742,0.7153762,0.7151872,0.7150067,0.714834,0.7146688,0.7145104,0.7143586,0.7142128,0.7140728,0.7139382,0.7138088,0.7136841,0.713564,0.7134482,0.7133364,0.7132286,0.7131244,0.7130237,0.7129263,0.7128321,0.7127408,0.7126525,0.7125668 };
        std::vector<floatingType> resultYShould1(std::begin(tmp), std::end(tmp));

        auto resultYIs0 = fData->y[0];
        auto resultYIs1 = fData->y[1];

        for (size_t i = 0; i < resultYIs0.size(); i++)
        {
            REQUIRE(std::abs(resultYIs0[i] - resultYShould0[i]) < tol);
            REQUIRE(std::abs(resultYIs1[i] - resultYShould1[i]) < tol);
        }
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualL1Aniso<floatingType>", "[flexProxDualL1Iso]")
{
    init2d();
    auto prox = new flexProxDualL1Aniso<floatingType>();
    prox->applyProx(1.0f, fData, { 0, 1 }, {});

    SECTION("prox result")
    {
        floatingType valueShould = 1;

        auto resultYIs0 = fData->y[0];
        auto resultYIs1 = fData->y[1];

        for (size_t i = 0; i < resultYIs0.size(); i++)
        {
            REQUIRE(std::abs(resultYIs0[i] - valueShould) < tol);
            REQUIRE(std::abs(resultYIs1[i] - valueShould) < tol);
        }
    }
    cleanup();
}



TEST_CASE("Prox: flexProxDualL2Inf<floatingType>", "[flexProxDualL2]")
{
    init2d();
    auto prox = new flexProxDualL2Inf<floatingType>();
    prox->applyProx(200.0f, fData, { 0, 1 }, {});


    SECTION("prox result")
    {
        std::vector<double> tmp = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.315943,1.30581,2.29607,3.28672,4.27773,5.26907,6.26073,7.2527,8.24495,9.23747,10.2302,11.2233,12.2165,13.21,14.2037,15.1976,16.1916 };
        std::vector<floatingType> resultYShould0(std::begin(tmp), std::end(tmp));
        tmp = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.322525,1.33246,2.342,3.35117,4.35999,5.36849,6.37667,7.38456,8.39218,9.39953,10.4066,11.4135,12.4201,13.4265,14.4328,15.4388,16.4446 };
        std::vector<floatingType> resultYShould1(std::begin(tmp), std::end(tmp));

        auto resultYIs0 = fData->y[0];
        auto resultYIs1 = fData->y[1];

        for (size_t i = 0; i < resultYIs0.size(); i++)
        {
            REQUIRE(std::abs(resultYIs0[i] - resultYShould0[i]) < tol);
            REQUIRE(std::abs(resultYIs1[i] - resultYShould1[i]) < tol);
        }
    }
    cleanup();
}

TEST_CASE("Prox: flexProxDualLInf<floatingType>", "[flexProxDualLInf]")
{
    init1d();
    auto prox = new flexProxDualLInf<floatingType>();
    prox->applyProx(30.0f, fData, { 0 }, {});


    SECTION("prox result")
    {
        std::vector<double> tmp = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,1.25,2.25,3.25,4.25,5.25,6.25,7.25 };
        std::vector<floatingType> resultYShould0(std::begin(tmp), std::end(tmp));

        auto resultYIs0 = fData->y[0];

        for (size_t i = 0; i < resultYIs0.size(); i++)
        {
            REQUIRE(std::abs(resultYIs0[i] - resultYShould0[i]) < tol);
        }
    }
    cleanup();
}
