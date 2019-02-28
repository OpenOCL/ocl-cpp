#include "EigenTensor.h"
#include <gtest/gtest.h>

TEST(testGeneralTensor, Constructor) {

  auto a = ocl::Tensor(4,3);

  auto b = ocl::Tensor(4,3);

  //auto c = a+b;

  ASSERT_EQ(6, 6);
}

TEST(testEigenTensor, Constructor) {

  auto T = ocl::Tensor(4,3);
  ASSERT_EQ(6, 6);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
