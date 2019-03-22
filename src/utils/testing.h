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

// Use the INFO macro to pass filename and line to the assertions
// e.g. ocl::test::assertEqual(4-1, 3, _INFO);
#define INFO ocl::test::Info(__FILE__, __LINE__)

namespace ocl
{
const std::string ansi_error = "\x1b[31m";
const std::string ansi_reset = "\x1b[0m";
const std::string ansi_highlight = "\x1b[36m";
const std::string ansi_emph = "\x1b[1m";

namespace test
{

// Struct carrying filename and line number
struct Info
{
  Info(std::string file, int line) : file(file), line(line) { }
  Info() : file("unknown"), line(-1) { }
  std::string file;
  int line;
};

static inline std::string toString(const std::vector<std::vector<int> >& vec)
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

static inline std::string toString(const std::vector<std::vector<double> >& vec)
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

static inline std::string toString(const std::vector<double>& vec)
{
  std::ostringstream str;
  str << "{";
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<double>(str, " "));
  str << "}";
  return str.str();
}

static inline std::string toString(const std::vector<int>& vec)
{
  std::ostringstream str;
  str << "{";
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(str, " "));
  str << "}";
  return str.str();
}

static inline void assertEqual(const int given, const int expected, const Info& info = Info())
{
  std::ostringstream str;
  str << std::endl;
  str << ocl::ansi_error << "Assertion failed "<< ocl::ansi_reset << "in file "
      << ocl::ansi_highlight << info.file << ocl::ansi_reset << " at line "
      << ocl::ansi_highlight << info.line << ocl::ansi_reset << std::endl;
  str << "Value should be " << expected << " but was " << given << std::endl;

  EXPECT_EQ(given, expected) << str.str();
}

static inline void assertEqual( const double given, const double expected,
                                const Info& info = Info(),
                                const double eps=1e-4)
{
  std::ostringstream str;
  str << std::endl;
  str << ocl::ansi_error << "Assertion failed "<< ocl::ansi_reset << "in file "
      << ocl::ansi_highlight << info.file << ocl::ansi_reset << " at line "
      << ocl::ansi_highlight << info.line << ocl::ansi_reset << std::endl;
  str << "Value should be " << expected << " but was " << given << std::endl;

  EXPECT_NEAR(given, expected, eps) << str.str();
}

static inline void assertEqualLength( const double length_given, const double length_expected,
                                      const Info& info, const std::string& given_str, const std::string& expected_str)
{
  std::ostringstream str;
  str << std::endl;
  str << ocl::ansi_error << "Assertion failed "<< ocl::ansi_reset << "in file "
      << ocl::ansi_highlight << info.file << ocl::ansi_reset << " at line "
      << ocl::ansi_highlight << info.line << ocl::ansi_reset << std::endl;
  str << "Vectors have different length, expected length is " << length_expected << " but was " << length_given << std::endl;
  str << "Expected vector: " << expected_str << ", given " <<  given_str << std::endl;

  ASSERT_EQ(length_given, length_expected) << str.str();
  // somehow gtest does not exit/fatal here..
  assert(0);
}

static inline void assertEqual( const std::vector<double>& given, const std::vector<double>& expected,
                                const Info& info = Info(),
                                const double eps = 1e-4)
{
  assertEqualLength(given.size(), expected.size(), info, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], info, eps);
  }
}

static inline void assertEqual( const std::vector<int>& given, const std::vector<int>& expected,
                                const Info& info = Info())
{
  assertEqualLength(given.size(), expected.size(), info, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], info);
  }
}

static inline void assertEqual( const std::vector<double>& given, const double expected,
                                const Info& info = Info(),
                                const double eps=1e-4)
{
  std::vector<double> exp_vec = {expected};
  assertEqual(given, exp_vec, info, eps);
}

static inline void assertEqual( const double given, const std::vector<double>& expected,
                                const Info& info = Info(),
                                const double eps=1e-4)
{
  std::vector<double> given_vec = {given};
  assertEqual(given_vec, expected, info, eps);
}

static inline void assertEqual( const std::vector<std::vector<int>>& given, const std::vector<std::vector<int>>& expected,
                                const Info& info = Info())
{
  assertEqualLength(given.size(), expected.size(), info, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], info);
  }
}

static inline void assertEqual( const std::vector<std::vector<double>>& given, const std::vector<std::vector<double>>& expected,
                                const Info& info = Info(),
                                const double eps=1e-4)
{
  assertEqualLength(given.size(), expected.size(), info, toString(given), toString(expected));
  for (unsigned int i = 0; i < expected.size(); ++i) {
    assertEqual(given[i], expected[i], info, eps);
  }
}

} // namespace test
} //namespace ocl
#endif // OCL_TESTING_H_
