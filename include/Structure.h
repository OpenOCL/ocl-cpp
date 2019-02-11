#ifndef OCPCPP_OCL_STRUCTURE_H_
#define OCPCPP_OCL_STRUCTURE_H_

#include <tuple>
#include <map>

#include "typedefs.h"

namespace ocl 
{

  struct ChildStructure
  {
    Structure structure;
    PositionArray positions;
  }

  class Structure
  {

  private:
    // length of the structure
    int len;
    // map to the children
    std::map<std::string, ChildStructure> children;

    // merges two position arrays
    static PositionArray merge(const PositionArray& p1, 
      const PositionArray& p2);

  public:
    Structure();
    ChildStructure get(const std::string& id, 
      const PositionArray& positions);
    Size size();

} // namespace ocl 


#endif  // OCPCPP_OCL_STRUCTURE_H_