

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
#ifndef OCL_UTILS_FUNCTIONS_H_
#define OCL_UTILS_FUNCTIONS_H_

namespace ocl {

// Array to vector.
static inline std::vector<int> toVector(const int* data, int nel)
{
  std::vector<int> values(data, data + nel);
  return values;
}

// Array to vector.
static inline std::vector<double> toVector(const double* data, int nel)
{
  std::vector<double> values(data, data + nel);
  return values;
}

// Concatenate two vectors
static inline std::vector<int> merge(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> s;
  s.insert( s.end(), a.begin(), a.end() );
  s.insert( s.end(), b.begin(), b.end() );
  return s;
}

// Product of all elements in vector
static inline int prod(const std::vector<int>& v) {
  int r = 1;
  for (unsigned int i=0; i<v.size(); i++) {
    r *= v[i];
  }
  return r;
}

} // namespace ocl
#endif // OCL_UTILS_FUNCTIONS_H_
