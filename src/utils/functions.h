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
#ifndef OCL_UTILS_FUNCTIONS_H_
#define OCL_UTILS_FUNCTIONS_H_

namespace ocl {

// End is included (closed interval)
static inline std::vector<int> linspace(int start, int end, int stride = 1) {
  std::vector<int> v;
  for(int idx = start; idx <= end; idx += stride) {
    v.push_back(idx);
  }
  return v;
}

static inline std::vector<double> linspace(double start, double end, double stride = 1.) {
  std::vector<double> v;
  for(double idx = start; idx <= end; idx += stride) {
    v.push_back(idx);
  }
  return v;
}

// End is not included (open interval)
static inline std::vector<int> range(int start, int end, int stride = 1) {
  std::vector<int> v;
  for(int idx = start; idx < end; idx += stride) {
    v.push_back(idx);
  }
  return v;
}

// Array to vector.
static inline std::vector<int> toVector(const int* data, int nel)
{
  std::vector<int> values(data, data + nel);
  return values;
}

// Array to vector.
static inline std::vector<int> toVector(const long long* data, int nel)
{
  std::vector<long long> values(data, data + nel);
  std::vector<int> r(values.begin(), values.end());
  return r;
}

// Array to vector.
static inline std::vector<double> toVector(const double* data, int nel)
{
  std::vector<double> values(data, data + nel);
  return values;
}

// Concatenate two vectors
template<class T>
static inline std::vector<T> merge(const std::vector<T>& a, const std::vector<T>& b)
{
  std::vector<T> s;
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
