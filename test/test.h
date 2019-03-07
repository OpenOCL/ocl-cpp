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

void assertDoubleFullEqual(const double* v1, const double v2) {
  ASSERT_EQ(v1[0],v2);
}

} // namespace test
} //namespace ocl
#endif // OCL_TEST_H_
