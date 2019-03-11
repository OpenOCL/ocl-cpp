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
#include "utils/functions.h"
#include "tensor/functions.h"

namespace ocl
{

// class TensorStructure : public Root {
//   TensorStructure(shape) : branches(Branches()), shape(shape), indizes(IndizesArray()) { }
// };

class Root
{
 public:

  Root(const std::map<std::string, Root>& branches,
       const std::vector<int>& nodeShape,
       const std::vector<std::vector<int> >& indizes)
      : branches(branches), nodeShape(nodeShape), indizes(indizes) { }

  bool hasBranches() const {
    return branches.empty();
  }

  // Return the shape of the nodes
  virtual std::vector<int> shape() const { return this->nodeShape; }
  // Returns the number of root nodes
  uint length() const { return indizes.size(); }

  // Return the complete shape including number of roots (length):
  // size := [shape length]
  std::vector<int> size() const
  {
    std::vector<int> s = this->shape();
    if (nodeShape.size() == 0 || this->length() > 1) {
      s = merge( s, {(int)this->length()} );
    }
    return s;
  }

  // Number of elements: prod(size)
  uint nel() const
  {
    std::vector<int> s = this->size();
    return std::accumulate(s.begin(), s.end(), 0);
  }

  Root get(std::string id) const
  {
    Root b = this->branches.at(id);
    IndizesArray idz = tensor::mergeArrays(indizes, b.indizes);
    return Root(b.branches, b.nodeShape, idz);
  }

  Root at(const int idx) const
  {
    std::vector<int> idz = {indizes[idx]};
    return Root(branches, nodeShape, idz);
  }

  Root slice(const std::vector<int>& slice1, const std::vector<int>& slice2) const
  {
    std::vector<std::vector<int> > a();
    for (uint i=0; i<indizes.size(); i++)
    {

      ::casadi::DM m_reshaped = ::casadi::DM(nodeShape.get(0), nodeShape.get(1));
      m_reshaped.set(indizes[i]);
      ::casadi::DM m_sliced = m_reshaped(slice1, slice2);

      // copy data to vector
      double *data = m_sliced.ptr();
      int nel = shape(m_sliced).numel();
      std::vector<double> values(data, data + nel);
      a.push_back(values);
    }
    return a;
  }



protected:
   // map to the children
   std::map<std::string, Root> branches;
   // length of the structure
   std::vector<int> nodeShape;
   // vector of indizes
   std::vector<std::vector<int> > indizes;
};



} // namespace ocl
#endif  // OCLCPP_OCL_ROOTNODE_H_
