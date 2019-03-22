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
#ifndef OCL_UTILS_ASSERTIONS_H_
#define OCL_UTILS_ASSERTIONS_H_

#include <stdlib.h>         // exit, EXIT_FAILURE
#include <ostream>

namespace ocl {

static inline void assertTrue(const bool expr, const std::string& msg)
{
  if (~expr) {
    std::cout << "Assertion failed: " << msg << std::endl;
  }
  exit(EXIT_FAILURE);
}

static inline void assertEqual(const int i, const int j, const std::string& msg)
{
  if (i!=j) {
    std::cout << "Assertion failed: " << msg << std::endl;
  }
  exit(EXIT_FAILURE);
}

} // namespace ocl
#endif // OCL_UTILS_ASSERTIONS_H_
