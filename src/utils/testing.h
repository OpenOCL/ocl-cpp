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
#ifndef OCL_TESTING_H_
#define OCL_TESTING_H_

#include <gtest/gtest.h>    // EXPECT_EQ, EXPECT_NEAR, ASSERT_EQ
#include <ostream>
#include <cstring>
#include <stdlib.h>         // exit, EXIT_FAILURE

#define DEF_FILENAME "unknown"

namespace ocl
{
namespace test
{

void assertEqual(const int given, const int expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME)
{
  std::ostringstream str;
  str << std::endl;
  if (std::strcmp(filename.c_str(), DEF_FILENAME) != 0) {
    str << "Assertion failed in file " << filename << " at line " << line_number << std::endl;
  }
  str << "Value should be " << expected << " but was " << given << std::endl;

  EXPECT_EQ(given, expected) << str.str();
}

void assertEqual(const double given, const double expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME,
                 const double eps=1e-4)
{
  std::ostringstream str;
  str << std::endl;
  if (std::strcmp(filename.c_str(), DEF_FILENAME) != 0) {
    str << "Assertion failed in file " << filename << " at line " << line_number << std::endl;
  }
  str << "Value should be " << expected << " but was " << given << std::endl;

  EXPECT_NEAR(given, expected, eps) << str.str();
}

void assertEqualLength(const int length_given, const int length_expected,
                       const int line_number = -1, const std::string& filename = DEF_FILENAME)
{
  std::ostringstream str;
  str << std::endl;
  if (std::strcmp(filename.c_str(), DEF_FILENAME) != 0) {
    str << "Assertion failed in file " << filename << " at line " << line_number << std::endl;
  }
  str << "Vectors have different length, expected length is " << length_expected << " but was " << length_given << std::endl;

  ASSERT_EQ(length_given, length_expected) << str.str();
  // somehow gtest does not exit/fatal here..
  exit(EXIT_FAILURE);
}

void assertEqual(const std::vector<double>& given, const std::vector<double>& expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME,
                 const double eps = 1e-4)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename);
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename, eps);
  }
}

void assertEqual(const std::vector<int>& given, const std::vector<int>& expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename);
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename);
  }
}

void assertEqual(const std::vector<double>& given, const double expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME,
                 const double eps=1e-4)
{
  std::vector<double> exp_vec = {expected};
  assertEqual(given, exp_vec, line_number, filename, eps);
}

void assertEqual(const double given, const std::vector<double>& expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME,
                 const double eps=1e-4)
{
  std::vector<double> given_vec = {given};
  assertEqual(given_vec, expected, line_number, filename, eps);
}

void assertEqual(const std::vector<std::vector<int>>& given, const std::vector<std::vector<int>>& expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename);
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename);
  }
}

void assertEqual(const std::vector<std::vector<double>>& given, const std::vector<std::vector<double>>& expected,
                 const int line_number = -1, const std::string& filename = DEF_FILENAME,
                 const double eps=1e-4)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename);
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename, eps);
  }
}

} // namespace test
} //namespace ocl
#endif // OCL_TESTING_H_
