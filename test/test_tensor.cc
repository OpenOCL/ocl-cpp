#include <utils/test.h>
#include "tensor/tensor.h"

TEST(testGeneralTensor, Slice) {

  auto a = ocl::Tensor::One(5,5)*10;
  auto r = ocl::slice(a, {3}, ocl::end(a, 2) );

  ocl::test::assertDoubleFullEqual( ocl::full(r), 10 );
}

TEST(testGeneralTensor, ScalarOperators) {
  // scalar unary operations
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::uplus(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 4 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::uminus(a);
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
    ocl::test::assertDoubleFullEqual( ocl::full(r), 1.38629436112 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sqrt(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 2 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::square(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 16 );
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sin(a);
    ocl::test::assertDoubleFullEqual( ocl::full(r), -0.7568024953 );
  }
  {
    auto a = ocl::Tensor(4);
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
    auto r = ocl::inverse(a);
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
  // {
  //   auto a = ocl::Tensor(0.44);
  //   auto r = ocl::asinh(a);
  //   ocl::test::assertDoubleFullEqual( ocl::full(r), 0.42691345412611653 );
  // }
  // {
  //   auto a = ocl::Tensor(2.2);
  //   auto r = ocl::acosh(a);
  //   ocl::test::assertDoubleFullEqual( ocl::full(r), 1.4254169430706127 );
  // }
  // {
  //   auto a = ocl::Tensor(0.22);
  //   auto r = ocl::atanh(a);
  //   ocl::test::assertDoubleFullEqual( ocl::full(r), 0.22365610902183242 );
  // }

  // binary operations
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::plus(a,b);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 4.43 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::minus(a,b);;
    ocl::test::assertDoubleFullEqual( ocl::full(r), -3.7699999999999996 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::times(a,b);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 1.353 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cdivide(a,b);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.0804878048780488 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cpow(a,b);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.010614686047848296 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cmin(a,b);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.33 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cmax(a,b);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 4.1 );
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::atan2(a,b);
    ocl::test::assertDoubleFullEqual( ocl::full(r), 0.08031466966032468 );
  }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
