/*
 *
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
#ifndef OCL_TEST_H_
#define OCL_TEST_H_

#include <gtest/gtest.h>

namespace ocl
{
namespace test
{

void assertDoubleFullEqual(const std::vector<double>& v1, const std::vector<double>& v2,
                           const double eps=1e-4) {

  EXPECT_EQ(v1.size(), v2.size()) << "Vectors v1 and v2 are of unequal length";
  for (unsigned int i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], v2[i], eps) << "Vectors v1 and v2 differ at index " << i;
  }
}

void assertDoubleFullEqual(const std::vector<double>& v1, const double v2,
                           const double eps=1e-4) {


  std::vector<double> v2vec = {v2};
  assertDoubleFullEqual(v1, v2vec, eps);
}

void assertEqual(const std::vector<double>& v1, const std::vector<double>& v2) {
  assertDoubleFullEqual(v1,v2);
}



} // namespace test
} //namespace ocl
#endif // OCL_TEST_H_