#ifndef OCPCPP_OCL_STRUCTURE_H_
#define OCPCPP_OCL_STRUCTURE_H_

#include <tuple>
#include <map>

#include "IntTensor.h"
#include "typedefs.h"

namespace ocl 
{

  class Structure
  {

  private:
    std::map<std::string, IntTensor> children;
    int len;

    IntTensor merge(const IntTensor& p1, const IntTensor& p2);

  public:
    Structure()
    std::tuple<Structure, IntTensor> get(const std::string& id, const IntTensor& positions);
    Size size();

} // namespace ocl 


#endif  // OCPCPP_OCL_STRUCTURE_H_