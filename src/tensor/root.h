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
#ifndef OCLCPP_OCL_ROOTNODE_H_
#define OCLCPP_OCL_ROOTNODE_H_

#include <tuple>
#include <map>

#include "typedefs.h"

namespace ocl
{

class Structure
{
  virtual const int length() = 0;
  virtual const int size() = 0;
  virtual const int nel() = 0;
};

class TensorStructure : public RootNode {
  TensorStructure(shape) : branches(Branches()), shape(shape), indizes(IndizesArray()) { }
};

class Root : public Structure
{
 public:

  Root(const Branches& branches, const Shape& shape, const IndizesArray& indizes)
      : branches(branches), shape(shape), indizes(indizes) { }

  boolean hasBranches() const {
    return branches.empty();
  }

  // Returns the number of root nodes
  uint length() const {
    return indizes.size();
  }

  Size size() const
  {
    Size s = shape;
    if (shape.size() == 0 || this->length() > 1) {
      s.append(this->length());
    }
    return s;
  }

  uint nel() const

  const RootNode get(const String& id, const IndizesArray& positions);

 private:
   // map to the children
   const Branches branches;
   // length of the structure
   const Shape shape;
   // vector of indizes
   const IndizesArray indizes;
};







int nel() const
{
  Size s = this->size;
  return std::accumulate(s.begin(), s.end(), 0);
}


RootNode get(const std::string& id) const
{
  auto b = branches[id];
  IndizesArray idz = Structure.mergeArrays(indizes, b.indizes);
  return RootNode(b.branches, b.shape, idz);
}

} // namespace ocl
#endif  // OCLCPP_OCL_ROOTNODE_H_
