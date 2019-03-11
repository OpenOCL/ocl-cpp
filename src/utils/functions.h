

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

static inline std::vector<int> merge(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> s;
  s.insert( s.end(), a.begin(), a.end() );
  s.insert( s.end(), b.begin(), b.end() );
  return s;
}

static inline int prod(const std::vector<int>& v) {
  int r = 1;
  for (unsigned int i=0; i<v.size(); i++) {
    r *= v[i];
  }
  return r;
}

static inline std::vector<int> linspace(int start, int end, int stride = 1) {
  std::vector<int> v;
  for(int idx = start; idx <= end; idx += stride) {
    v.push_back(idx);
  }
  return v;
}


} // namespace ocl
#endif // OCL_UTILS_FUNCTIONS_H_
