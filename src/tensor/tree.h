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
#ifndef OCLCPP_OCL_TREE_H_
#define OCLCPP_OCL_TREE_H_

#include <tuple>
#include <map>  // branches

#include "utils/typedefs.h"
#include "utils/functions.h"   // prod
#include "tensor/functions.h"  // tensor::range
#include "tensor/casadi.h"     // slice

// This file defines classes Tree and Leaf
namespace ocl
{
// Tree structure with children accessable by id
// The tree can have multiple roots (number given by length, number of vectors
// in indizes), each root has the same shape given by nodeShape
class Tree
{
 public:

  Tree(const std::map<std::string, Tree>& branches,
       const std::vector<int>& nodeShape,
       const std::vector<std::vector<int> >& indizes)
      : branches(branches), nodeShape(nodeShape), indizes(indizes) { }

  // Check if there are subtrees
  bool hasBranches() const {
    return branches.empty();
  }

  // get indizes of trajectory element i
  std::vector<int> indizes(int i) {
    return indizes[i];
  }

  // Return the shape of the nodes
  virtual std::vector<int> shape() const { return this->nodeShape; }

  // Returns the number of root nodes
  uint length() const { return indizes.size(); }

  // Returns total number of elements
  int size() const
  {
    std::vector<int> s = this->shape();
    return prod(s) * this->length;
  }

  // Get subtree by string id
  Tree get(const std::string& id) const
  {
    Tree b = this->branches.at(id);
    std::vector<std::vector<int> > idz = tensor::mergeIndizes(indizes, b.indizes);
    return Tree(b.branches, b.nodeShape, idz);
  }

  // Cut tree, get single element of trajectory
  Tree at(const int idx) const
  {
    std::vector<std::vector<int> > idz = {indizes[idx]};
    return Tree(branches, nodeShape, idz);
  }

  // Slice matrizes in trajectory
  Tree slice(const std::vector<int>& slice1, const std::vector<int>& slice2) const
  {
    std::vector<std::vector<int> > a;
    for (uint i=0; i<indizes.size(); i++)
    {

      ::casadi::IM m_reshaped = ::casadi::IM(nodeShape[0], nodeShape[1]);
      m_reshaped.set( indizes[i], false, linspace(0, indizes[i].size()) );
      ::casadi::IM m_sliced = m_reshaped(slice1, slice2);

      // copy data to vector
      int *data = m_sliced.ptr();
      int nel = m_sliced.size1()*m_sliced.size2();
      a.push_back(toVector(data, nel));

      std::vector<int> sliceShape;
    }
    std::map<std::string, Tree> branches;
    return Tree(branches, sliceShape, a);
  }

private:
   // map to the children
   std::map<std::string, Tree> branches;
   // length of the structure
   std::vector<int> nodeShape;
   // vector of indizes
   std::vector<std::vector<int> > indizes;
};

// A tree with no children/branches
class Leaf : public Tree {
  Leaf(const std::vector<int>& shape)
    : branches(Branches()), shape(shape),
      indizes( {tensor::range(0, prod(shape))} ) { }
};

} // namespace ocl
#endif  // OCLCPP_OCL_TREE_H_
