#include "EigenTensor.h"
#include <gtest/gtest.h>
 
TEST(testEigenTensor, Constructor) { 

  auto T = ocl::EigenTensor<2>({4,3});
  ASSERT_EQ(6, 6);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}