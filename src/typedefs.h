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
#ifndef OCLCPP_OCL_TYPEDEFS_H_
#define OCLCPP_OCL_TYPEDEFS_H_

namespace ocl
{
typedef float Scalar;
typedef int Integer;
typedef std::vector<int> IndizesArray;
typedef std::string String;
typedef unsigned int uint;

typedef std::map<const std::string&, const RootNode> Branches;

class Shape
{
public:
  Shape(std::initializer_list<int> s) :s(s) { }

  int get(int i) {
    return s[i];
  }

  int numel() {
    int r = 0;
    for (unsigned int i=0;i<s.size();i++) {
      r *= s[i];
    }
    return r;
  }

private:
  const std::vector<int> s;
};

}
#endif // OCLCPP_OCL_TYPEDEFS_H_
