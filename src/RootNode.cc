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
#include "RootNode.h"

#include <numeric>

namespace ocl
{



  RootNode(const Branches branches, const Shape shape, const IndizesArray indizes)
      : branches(branches), shape(shape), indizes(indizes)
  {
  }

  boolean hasBranches() const
  {
    return branches.empty();
  }

  int length() const
  {
    return indizes.size();
  }

  Size size() const
  {
    Size s = shape;
    if (shape.size() == 0 || this->length() > 1)
    {
      s.append(this->length());
    }
    return s;
  }

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


  static IndizesArray RootNode::mergeArrays(const IndizesArray &p1,
      const IndizesArray &p2);
  {
    // Combine arrays of positions
    // p2 are relative to p1
    // Returns: absolute p2
    const auto& s1 = p1.size();
    const auto& s2 = p2.size();

    IndizesArray pout(s2[0].size(),s1*s2);
    for(int k=0; k<s1; k++)
    {
      auto ap1 = p1[k];

      for(int l=0; l<s2; l++)
      {
        auto ap2 = p2[l];
        pout[l+(k-1)*K2] = ap1[ap2];
      }
    }

  } // mergeArrays

} // namespace ocl
