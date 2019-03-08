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
#ifndef OCL_TENSOR_SHAPE_H_
#define OCL_TENSOR_SHAPE_H_

namespace ocl {

class Shape
{
public:
  Shape(int el) :s({el}) { }
  Shape(const std::vector<int>& s) :s(s) { }
  Shape(std::initializer_list<int> s) :s(s) { }

  int get(int i) const { return s[i]; }
  std::vector<int> vec() const { return this->s; }

  int numel() const {
    int r = 1;
    for (unsigned int i=0;i<s.size();i++) {
      r *= s[i];
    }
    return r;
  }

private:
  const std::vector<int> s;
};

static inline Shape merge(const Shape& a, const Shape& b)
{
  std::vector<int> s;
  s.insert( s.end(), a.vec().begin(), a.vec().end() );
  s.insert( s.end(), b.vec().begin(), b.vec().end() );
  return Shape(s);
}

} // namespace ocl
#endif // OCL_TENSOR_SHAPE_H_
