
#include "Structure.h"

namespace ocl
{

  static PositionArray Structure::merge(const PositionArray &p1, 
      const PositionArray &p2);
  {
    // Combine arrays of positions
    // p2 are relative to p1
    // Returns: absolute p2
    const auto& s1 = p1.size();
    const auto& s2 = p2.size();

    PositionArray pout(s2[0].size(),s1*s2);
    for(int k=0; k<s1; k++)
    {
      auto ap1 = p1[k];

      for(int l=0; l<s2; l++)
      {
        auto ap2 = p2[l];
        pout[l+(k-1)*K2] = ap1[ap2];
      }
    } 

  } // merge

  Structure()
  {
    len = 0;
    children = std::map<std::string, ChildStructure>();
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

} // namespace ocl