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
  virtual Shape shape() = 0;
  virtual uint length() = 0;
  virtual Shape size() = 0;
  virtual uint nel() = 0;
  virtual Structure get(const std::string& id);
  virtual Structure get(const int idx);
  virtual Structure slice(const Slices& slices);
};

class TensorStructure : public Root {
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
  virtual Shape shape() const override { return this->nodeShape; }
  // Returns the number of root nodes
  uint length() const override { return indizes.size(); }

  // Return the complete shape including number of roots (length):
  // size := [shape length]
  Shape size() const override
  {
    Shape s = this->shape();
    if (shape.size() == 0 || this->length() > 1) {
      s = merge( s, Shape(this->length()) );
    }
    return s;
  }

  // Number of elements: prod(size)
  uint nel() const override
  {
    Shape s = this->size();
    return std::accumulate(s.begin(), s.end(), 0);
  }

  virtual Root get(const std::string& id) const override
  {
    auto b = branches[id];
    IndizesArray idz = Structure.mergeArrays(indizes, b.indizes);
    return Root(b.branches, b.nodeShape, idz);
  }

  virtual Root at(const int idx) override
  {
    std::vector<int> idz = {indizes[idx]};
    return Root(branches, nodeShape, idz);
  }

  virtual Root block(const int i, const int j, const int k, const int l)
  {
    IndizesArray a();
    for (uint i=0; i<indizes.size(); i++)
    {
      // Pass indizes memory to eigen to reshape and slice indizes
      std::vector<int> idz = indizes[i];
      int* idz_ptr = &idz[0];
      eigen::Map<const eigen::Array<int,Dynamic,Dynamic> > m_reshaped(idz_ptr, nodeShape.get(0), nodeShape.get(1));
      eigen::Array<int,Dynamic,Dynamic> m_sliced = m_reshaped.block(i, j, k, l);

      // copy data to vector
      std::vector<int> dest(m_sliced.size);
      dest.assign(m_sliced.data(), m_sliced.data() + m_sliced.size());
      a.push_back(dest);
    }
    return a;
  }



protected:
   // map to the children
   const Branches branches;
   // length of the structure
   const Shape nodeShape;
   // vector of indizes
   const IndizesArray indizes;
};



} // namespace ocl
#endif  // OCLCPP_OCL_ROOTNODE_H_
