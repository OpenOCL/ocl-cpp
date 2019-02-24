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

namespace ocl
{



  RootNode(const Branches branches, const Shape shape, const IndizesArray indizes)
      : branches(branches), shape(shape), indizes(indizes)
  {
  }

  const boolean hasBranches() const
  {
    
  }


  ChildStructure get(const std::string& id,
      const PositionArray& positions)
  {
    child = children[id];

    c = ChildStructure();
    c.positions = Structure.merge(positions,child.positions);
    c.structure = child.structure;
    return c;
  }

  Size size()
  {
    return Size({len});
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
