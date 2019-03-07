#include <utils/test.h>
#include "tree_tensor/tensor.h"

TEST(testGeneralTensor, Constructor) {

  ocl::Tensor a = ocl::Tensor::Zeros(4,3);
  ocl::Tensor b = ocl::Tensor::Ones(4,3);

  ocl::cos(b).disp();
  b.cos().disp();
  b.disp();

  ASSERT_EQ(6, 6);
}

TEST(testCasadiTensor, ScalarOperators) {
  // scalar unary operations
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::uplus(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 4 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::negate(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), -4 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::exp(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 54.5981500331 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::log(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.60205999132 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sqrt(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 2 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sq(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 16 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sin(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), -0.7568024953 );
  }
  {
    auto a = ocl::Tensor::Fill(4);
    auto r = ocl::sin(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), -0.7568024953 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::cos(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), -0.65364362086 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::tan(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 1.15782128235 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::asin(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.4555986733958234 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::acos(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 1.1151976533990733 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::atan(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.41450687458478597 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::abs(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.44 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::inv(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 2.272727272727273 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::sinh(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.45433539871409734 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::cosh(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 1.0983718197972387 );
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::asinh(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.42691345412611653 );
  }
  {
    auto a = ocl::Tensor(2.2);
    auto r = ocl::acosh(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 1.4254169430706127 );
  }
  {
    auto a = ocl::Tensor(0.22);
    auto r = ocl::atanh(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.22365610902183242 );
  }

  // binary operations
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::plus(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 4.43 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::minus(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), -3.7699999999999996 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::times(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 1.353 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::divide(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.0804878048780488 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::pow(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.010614686047848296 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::min(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.33 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::max(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 4.1 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::atan2(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.08031466966032468 );
  }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
