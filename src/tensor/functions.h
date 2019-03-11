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
#ifndef OCL_TENSOR_FUNCTIONS_H_
#define OCL_TENSOR_FUNCTIONS_H_

namespace ocl
{

namespace tensor
{

static std::vector<std::vector<int> > mergeIndizes(
    const std::vector<std::vector<int> >& p1,
    const std::vector<std::vector<int> >& p2)
{
  // Combine arrays of positions
  // p2 are relative to p1
  // Returns: absolute p2
  int s1 = p1.size();
  int s2 = p2.size();

  std::vector<std::vector<int> > pout(p2[0].size(),s1*s2);
  for(int k=0; k<s1; k++)
  {
    std::vector<int> ap1 = p1[k];
    for(int l=0; l<s2; l++)
    {
      std::vector<int> ap2 = p2[l];
      pout[l+(k-1)*K2] = ap1[ap2];
    }
  }
} // mergeArrays

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

} // namespace tensor
} // namespace ocl
#endif // OCL_TENSOR_FUNCTIONS_H_
