#include <gtest/gtest.h>

#include "Tensor.h"
#include "SymbolicAdMatrix.h"


TEST(testGeneralTensor, Constructor) {

  ocl::Tensor<ocl::NumericMatrix> a = ocl::Tensor<ocl::NumericMatrix>(4,3);

  ocl::Tensor<ocl::SymbolicAdMatrix> b = ocl::Tensor<ocl::SymbolicAdMatrix>(4,3);

  ocl::cos(a).disp();
  a.cos().disp();

  b.disp();

  auto c = ocl::Tensor<ocl::NumericMatrix>(4,3);

  //auto c = a+b;

  ASSERT_EQ(6, 6);
}

TEST(testEigenTensor, Constructor) {

  auto T = ocl::Tensor<ocl::NumericMatrix>(4,3);
  ASSERT_EQ(6, 6);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
