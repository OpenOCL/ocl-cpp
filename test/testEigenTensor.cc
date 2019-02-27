#include "EigenTensor.h"
#include <gtest/gtest.h>

TEST(testGeneralTensor, Constructor) {

  auto a = ocl::Tensor<2>({4,3});
  a.set({{1,2,3,4},{2,3,4,5},{3,4,5,6}});

  auto b = ocl::Tensor<2>({4,3});
  b.set({{3}});

  auto c = a+b;

  ASSERT_EQ(6, 6);
}

TEST(testEigenTensor, Constructor) {

  auto T = ocl::Tensor<2>({4,3});
  ASSERT_EQ(6, 6);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
