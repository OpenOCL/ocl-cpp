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

#define DEFAULT_FILENAME "unknown"
#define ANSI_ERROR "\x1b[31m"
#define ANSI_RESET "\x1b[0m"
#define ANSI_HIGHLIGHT "\x1b[36m"
#define ANSI_EMPH "\x1b[1m"

namespace ocl
{
namespace test
{

std::string toString(const std::vector<std::vector<int> >& vec)
{
  std::ostringstream str;
  str << "{";
  for (unsigned int i = 0; i < vec.size(); ++i)
  {
    str << "{";
    std::copy(vec[i].begin(), vec[i].end(), std::ostream_iterator<int>(str, " "));
    str << "}";
  }
  str << "}";
  return str.str();
}

std::string toString(const std::vector<std::vector<double> >& vec)
{
  std::ostringstream str;
  str << "{";
  for (unsigned int i = 0; i < vec.size(); ++i)
  {
    str << "{";
    std::copy(vec[i].begin(), vec[i].end(), std::ostream_iterator<double>(str, " "));
    str << "}";
  }
  str << "}";
  return str.str();
}

std::string toString(const std::vector<double>& vec)
{
  std::ostringstream str;
  str << "{";
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<double>(str, " "));
  str << "}";
  return str.str();
}

std::string toString(const std::vector<int>& vec)
{
  std::ostringstream str;
  str << "{";
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(str, " "));
  str << "}";
  return str.str();
}

void assertEqual(const int given, const int expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME)
{
  std::ostringstream str;
  str << std::endl;
  if (std::strcmp(filename.c_str(), DEFAULT_FILENAME) != 0) {
    str << ANSI_ERROR << "Assertion failed "<< ANSI_RESET << "in file " << ANSI_HIGHLIGHT << filename << ANSI_RESET << " at line " << ANSI_HIGHLIGHT << line_number << ANSI_RESET << std::endl;
  }
  str << "Value should be " << expected << " but was " << given << std::endl;

  EXPECT_EQ(given, expected) << str.str();
}

void assertEqual(const double given, const double expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME,
                 const double eps=1e-4)
{
  std::ostringstream str;
  str << std::endl;
  if (std::strcmp(filename.c_str(), DEFAULT_FILENAME) != 0) {
        str << ANSI_ERROR << "Assertion failed "<< ANSI_RESET << "in file " << ANSI_HIGHLIGHT << filename << ANSI_RESET << " at line " << ANSI_HIGHLIGHT << line_number << ANSI_RESET << std::endl;
  }
  str << "Value should be " << expected << " but was " << given << std::endl;

  EXPECT_NEAR(given, expected, eps) << str.str();
}

void assertEqualLength(const int length_given, const int length_expected,
                       const int line_number = -1, const std::string& filename = DEFAULT_FILENAME,
                       const std::string& given_str = "", const std::string& expected_str = "")
{
  std::ostringstream str;
  str << std::endl;
  if (std::strcmp(filename.c_str(), DEFAULT_FILENAME) != 0) {
        str << ANSI_ERROR << "Assertion failed "<< ANSI_RESET << "in file " << ANSI_HIGHLIGHT << filename << ANSI_RESET << " at line " << ANSI_HIGHLIGHT << line_number << ANSI_RESET << std::endl;
  }
  str << "Vectors have different length, expected length is " << length_expected << " but was " << length_given << std::endl;
  str << "Expected vector: " << expected_str << ", given " <<  given_str << std::endl;

  ASSERT_EQ(length_given, length_expected) << str.str();
  // somehow gtest does not exit/fatal here..
  assert(0);
}

void assertEqual(const std::vector<double>& given, const std::vector<double>& expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME,
                 const double eps = 1e-4)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename, eps);
  }
}

void assertEqual(const std::vector<int>& given, const std::vector<int>& expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename);
  }
}

void assertEqual(const std::vector<double>& given, const double expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME,
                 const double eps=1e-4)
{
  std::vector<double> exp_vec = {expected};
  assertEqual(given, exp_vec, line_number, filename, eps);
}

void assertEqual(const double given, const std::vector<double>& expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME,
                 const double eps=1e-4)
{
  std::vector<double> given_vec = {given};
  assertEqual(given_vec, expected, line_number, filename, eps);
}

void assertEqual(const std::vector<std::vector<int>>& given, const std::vector<std::vector<int>>& expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename);
  }
}

void assertEqual(const std::vector<std::vector<double>>& given, const std::vector<std::vector<double>>& expected,
                 const int line_number = -1, const std::string& filename = DEFAULT_FILENAME,
                 const double eps=1e-4)
{
  assertEqualLength(given.size(), expected.size(), line_number, filename, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], line_number, filename, eps);
  }
}

} // namespace test
} //namespace ocl
#endif // OCL_TESTING_H_
