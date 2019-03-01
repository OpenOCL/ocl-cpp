#include <gtest/gtest.h>

#include "Tensor.h"
#include "NumericMatrix.h"


TEST(testGeneralTensor, Constructor) {

  ocl::Tensor<ocl::NumericMatrix> a = ocl::Tensor<ocl::NumericMatrix>(4,3);
  ocl::cos(a);
  a.cos();

  auto b = ocl::Tensor<ocl::NumericMatrix>(4,3);

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
