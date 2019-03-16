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

// TreeBuilder is a friend class so that it can construct trees
friend class TreeBuilder;

public:

  // constructor for empty branches
  static std::map<std::string, Tree> Branches() {
    return std::map<std::string, Tree>();
  }

  Tree() : _branches(std::map<std::string, Tree>()), _shape({}), _indizes({}) { }

  Tree(const std::map<std::string, Tree>& branches,
       const std::vector<int>& shape,
       const std::vector<std::vector<int> >& indizes)
      : _branches(branches), _shape(shape), _indizes(indizes) { }

  // Check if there are subtrees
  bool hasBranches() const {
    return this->_branches.empty();
  }

  // get indizes of trajectory element i
  std::vector<int> getIndizes(int i) {
    return this->_indizes[i];
  }

  // Return the shape of the nodes
  std::vector<int> shape() const {
    return this->_shape;
  }

  // Return indizes vector
  std::vector<std::vector<int> > indizes() {
    return this->_indizes;
  }

  // Returns the number of root nodes
  uint length() const { return this->_indizes.size(); }

  // Returns total number of elements
  int size() const
  {
    std::vector<int> s = this->shape();
    return prod(s) * this->length();
  }

  // Get subtree by string id
  Tree get(const std::string& id) const
  {
    Tree b = this->_branches.at(id);
    std::vector<std::vector<int> > idz = tensor::mergeIndizes(this->_indizes, b._indizes);
    return Tree(b._branches, b._shape, idz);
  }

  // Cut tree, get single element of trajectory
  Tree at(const int idx) const
  {
    std::vector<std::vector<int> > idz = {this->_indizes[idx]};
    return Tree(this->_branches, this->_shape, idz);
  }

  // Slice matrizes in trajectory
  Tree slice(const std::vector<int>& slice1, const std::vector<int>& slice2) const
  {
    std::vector<std::vector<int> > a;
    for (uint i=0; i<this->_indizes.size(); i++)
    {

      ::casadi::IM m_reshaped = ::casadi::IM(this->_shape[0], this->_shape[1]);
      m_reshaped.set( this->_indizes[i], false, linspace(0, this->_indizes[i].size()) );
      ::casadi::IM m_sliced = m_reshaped(slice1, slice2);

      // copy data to vector
      int *data = m_sliced.ptr();
      int nel = m_sliced.size1()*m_sliced.size2();
      a.push_back(toVector(data, nel));
    }
    std::vector<int> sliceShape {(int)slice1.size(), (int)slice2.size()};
    std::map<std::string, Tree> branches;
    return Tree(branches, sliceShape, a);
  }

private:
   // map to the children
   std::map<std::string, Tree> _branches;
   // length of the structure
   std::vector<int> _shape;
   // vector of indizes
   std::vector<std::vector<int> > _indizes;
};

// A tree with no children/branches
class Leaf : public Tree {
  Leaf(const std::vector<int>& shape)
    : Tree(std::map<std::string, Tree>(), shape, {range(0, prod(shape))}) { }
};

} // namespace ocl
#endif  // OCLCPP_OCL_TREE_H_
