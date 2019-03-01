
#include "Tensor.h"

template<class M>
static inline Tensor<M> ocl::cos(const Tensor<M>& t) { return Tensor<M>::unaryVecOperation(t, &M::cos); }
