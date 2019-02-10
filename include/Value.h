#ifndef OCPCPP_OCL_NUMERICVALUE_H_
#define OCPCPP_OCL_NUMERICVALUE_H_

#include "IntTensor.h"

namespace ocl 
{

  class NumericValue
  {

  private:
    float[] value;
    IntTensor merge(IntTensor p1, IntTensor p2);

  public:
    NumericValue(float[] value);
    int numel();
    void set(Structure type, IntTensor positions, float[] value);
    float[] value(IntTensor positions);  


} // namespace ocl 


#endif  // OCPCPP_OCL_NUMERICVALUE_H_