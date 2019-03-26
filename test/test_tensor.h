/*
 *    Copyright (C) 2019 Jonas Koenemann
 *
 *    This program is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 */
#include <utils/testing.h>
#include "tensor/tensor.h"

TEST(Tensor, aSlice) {

  auto a = ocl::Tensor::One(5,5)*10;
  auto r = ocl::slice(a, {2}, ocl::end(a, 1));

  ocl::test::assertEqual( ocl::full(r), {{10}}, OCL_INFO);
}

TEST(Tensor, bScalarOperators) {
  // scalar unary operations
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::uplus(a);
    ocl::test::assertEqual( ocl::full(r), {{4}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::uminus(a);
    ocl::test::assertEqual( ocl::full(r), {{-4}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::exp(a);
    ocl::test::assertEqual( ocl::full(r), {{54.5981500331}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::log(a);
    ocl::test::assertEqual( ocl::full(r), {{1.38629436112}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sqrt(a);
    ocl::test::assertEqual( ocl::full(r), {{2}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::square(a);
    ocl::test::assertEqual( ocl::full(r), {{16}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sin(a);
    ocl::test::assertEqual( ocl::full(r), {{-0.7568024953}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::sin(a);
    ocl::test::assertEqual( ocl::full(r), {{-0.7568024953}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::cos(a);
    ocl::test::assertEqual( ocl::full(r), {{-0.65364362086}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(4);
    auto r = ocl::tan(a);
    ocl::test::assertEqual( ocl::full(r), {{1.15782128235}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::asin(a);
    ocl::test::assertEqual( ocl::full(r), {{0.4555986733958234}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::acos(a);
    ocl::test::assertEqual( ocl::full(r), {{1.1151976533990733}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::atan(a);
    ocl::test::assertEqual( ocl::full(r), {{0.41450687458478597}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::abs(a);
    ocl::test::assertEqual( ocl::full(r), {{0.44}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::inverse(a);
    ocl::test::assertEqual( ocl::full(r), {{2.272727272727273}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::sinh(a);
    ocl::test::assertEqual( ocl::full(r), {{0.45433539871409734}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.44);
    auto r = ocl::cosh(a);
    ocl::test::assertEqual( ocl::full(r), {{1.0983718197972387}}, OCL_INFO);
  }
  // {
  //   auto a = ocl::Tensor(0.44);
  //   auto r = ocl::asinh(a);
  //   ocl::test::assertEqual( ocl::full(r), 0.42691345412611653, OCL_INFO);
  // }
  // {
  //   auto a = ocl::Tensor(2.2);
  //   auto r = ocl::acosh(a);
  //   ocl::test::assertEqual( ocl::full(r), 1.4254169430706127, OCL_INFO);
  // }
  // {
  //   auto a = ocl::Tensor(0.22);
  //   auto r = ocl::atanh(a);
  //   ocl::test::assertEqual( ocl::full(r), 0.22365610902183242, OCL_INFO);
  // }

  // binary operations
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::plus(a,b);
    ocl::test::assertEqual( ocl::full(r), {{4.43}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::minus(a,b);;
    ocl::test::assertEqual( ocl::full(r), {{-3.7699999999999996}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::times(a,b);
    ocl::test::assertEqual( ocl::full(r), {{1.353}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cdivide(a,b);
    ocl::test::assertEqual( ocl::full(r), {{0.0804878048780488}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cpow(a,b);
    ocl::test::assertEqual( ocl::full(r), {{0.010614686047848296}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cmin(a,b);
    ocl::test::assertEqual( ocl::full(r), {{0.33}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::cmax(a,b);
    ocl::test::assertEqual( ocl::full(r), {{4.1}}, OCL_INFO);
  }
  {
    auto a = ocl::Tensor(0.33);
    auto b = ocl::Tensor(4.1);
    auto r = ocl::atan2(a,b);
    ocl::test::assertEqual( ocl::full(r), {{0.08031466966032468}}, OCL_INFO);
  }
}
