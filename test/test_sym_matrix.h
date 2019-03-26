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
#include "tensor/matrix.h"

TEST(SymMatrix, aScalarOperators)
{
  // scalar unary operations
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::uplus(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), 4, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::uminus(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), -4, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::exp(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), 54.5981500331, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::log(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), 1.38629436112, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::sqrt(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), 2, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::square(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), 16, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::sin(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), -0.7568024953, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::cos(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), -0.65364362086, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::tan(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {4}), 1.15782128235, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::asin(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {0.44}), 0.4555986733958234, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::acos(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {0.44}), 1.1151976533990733, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::atan(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {0.44}), 0.41450687458478597, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::abs(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {0.44}), 0.44, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::inverse(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {0.44}), 2.272727272727273, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::sinh(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {0.44}), 0.45433539871409734, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto r = ocl::cosh(a);
    ocl::test::assertEqual( ocl::full(r, {a}, {0.44}), 1.0983718197972387, OCL_INFO);
  }

  // binary operations
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::plus(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), 4.43, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::minus(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), -3.7699999999999996, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::ctimes(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), 1.353, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::cdivide(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), 0.0804878048780488, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::cpow(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), 0.010614686047848296, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::cmin(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), 0.33, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::cmax(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), 4.1, OCL_INFO);
  }
  {
    auto a = ocl::Matrix::Sym(1,1);
    auto b = ocl::Matrix::Sym(1,1);
    auto r = ocl::atan2(a,b);
    ocl::test::assertEqual( ocl::full(r, {a,b}, {0.33, 4.1}), 0.08031466966032468, OCL_INFO);
  }
}
