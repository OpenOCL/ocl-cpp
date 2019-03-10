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
#ifndef OCL_TENSOR_SLICE_H_
#define OCL_TENSOR_SLICE_H_

namespace ocl {

class Slicable
{
public:
  virtual int size(int dim) const = 0;
};

static inline std::vector<int> linspace(int start, int end, int stride = 1) {
  std::vector<int> v;
  for(int idx = start; idx <= end; idx += stride) {
    v.push_back(idx);
  }
  return v;
}

static inline std::vector<int> all(const Slicable& obj, const int dim) {
 return linspace(0, obj.size(dim)-1);
}

static inline std::vector<int> end(const Slicable& obj, const int dim) {
 return {obj.size(dim)-1};
}

} // namespace ocl
#endif // OCL_TENSOR_SLICE_H_
