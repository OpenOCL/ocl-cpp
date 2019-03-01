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
 *    You should have received a copy of the GNU General Public
 *    License along with this program;
 *    if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */
#ifndef OCLCPP_OCL_ROOTNODE_H_
#define OCLCPP_OCL_ROOTNODE_H_

#include <tuple>
#include <map>

#include "typedefs.h"

namespace ocl
{

typedef std::map<const std::string&, const RootNode> Branches;
typedef std::vector<int> Shape;
typedef std::vector<std::vector<int>> IndizesArray;

static RootNode flattenTree(const RootNode& tree)
{
  TreeBuilder tout = TreeBuilder();
  flattenTreeIterate(tree, &tout);
  return tout;
}

static void flattenTreeIterate(const RootNode& tree, RootNode* tout)
{
  for (const auto& b : tree.branches) {
    RootNode child = tree.get(b.first);
    if (child.hasBranches()) {
      flattenTreeIterate(child, tout);
    } else {
      tout->add(b, child);
    }
  } // end for
}

class Structure
{
  virtual const int length() = 0;
  virtual const int size() = 0;
  virtual const int nel() = 0;
};

class TensorStructure : public RootNode {
  TensorStructure(shape) : branches(Branches()), shape(shape), indizes(IndizesArray()) { }
};

class RootNode : public Structure
{
 public:

  RootNode(const Branches branches, const Shape shape, const IndizesArray indizes);

  const boolean hasBranches() const;

  const int length() const;

  const Shape size() const;

  const int nel() const

  const RootNode get(const String& id, const IndizesArray& positions);

  // merges two position arrays
  static IndizesArray mergeArrays(const IndizesArray& p1,
    const IndizesArray& p2);

 private:
   // map to the children
   const Branches branches;
   // length of the structure
   const Shape shape;
   // vector of indizes
   const IndizesArray indizes;
};

} // namespace ocl
#endif  // OCLCPP_OCL_ROOTNODE_H_
