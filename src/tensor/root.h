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

#include "utils/typedefs.h"

namespace ocl
{

class Structure
{
  virtual uint length() = 0;
  virtual Size size() = 0;
  virtual uint nel() = 0;
};

class TensorStructure : public RootNode {
  TensorStructure(shape) : branches(Branches()), shape(shape), indizes(IndizesArray()) { }
};

class Root : public Structure
{
 public:

  typedef std::map<const std::string&, const RootNode> Branches;
  typedef std::vector<int> IndizesArray;

  Root(const Branches& branches, const Shape& shape, const IndizesArray& indizes)
      : branches(branches), shape(shape), indizes(indizes) { }
 
  boolean hasBranches() const {
    return branches.empty();
  }

  // Return the shape of the nodes
  Size shape() const { return this->nodeShape; }
  // Returns the number of root nodes
  uint length() const { return indizes.size(); }

  // Return the complete shape including number of roots (length):
  // size := [shape length]
  Shape size() const
  {
    Shape s = this->shape();
    if (shape.size() == 0 || this->length() > 1) {
      s = merge( s, Shape(this->length()) );
    }
    return s;
  }

  // Number of elements: prod(size)
  uint nel() const
  {
    Shape s = this->size();
    return std::accumulate(s.begin(), s.end(), 0);
  }

  const RootNode get(const String& id, const IndizesArray& positions);

 private:
   // map to the children
   const Branches branches;
   // length of the structure
   const Shape nodeShape;
   // vector of indizes
   const IndizesArray indizes;
};

RootNode get(const std::string& id) const
{
  auto b = branches[id];
  IndizesArray idz = Structure.mergeArrays(indizes, b.indizes);
  return RootNode(b.branches, b.shape, idz);
}

} // namespace ocl
#endif  // OCLCPP_OCL_ROOTNODE_H_
