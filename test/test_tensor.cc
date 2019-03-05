#include <gtest/gtest.h>
#include "tree_tensor/tensor.h"


TEST(testGeneralTensor, Constructor) {

  ocl::Tensor a = ocl::Tensor(4,3);
  ocl::Tensor b = ocl::Tensor(4,3);

  ocl::cos(a).disp();
  a.cos().disp();

  b.disp();

  auto c = ocl::Tensor(4,3);

  //auto c = a+b;

  ASSERT_EQ(6, 6);
}

TEST(testEigenTensor, Constructor) {

  auto T = ocl::Tensor(4,3);
  ASSERT_EQ(6, 6);
}


// int main(int argc, char **argv) {
//     testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
