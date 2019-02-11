#include "Value.h"

namespace ocl {

  template<class T>
  static Value<T> Value::create(const Structure &structure, 
      const T &value)
  {
    s = structure->size();
    PositionArray positions(s);
    positions = value;
    oclValue = Value<T>(structure, positions, value);
  }

  // constructor
  template<class T>
  Value(const Structure &structure, const PositionArray &positions, const T &value)
  {
    structure = structure;
    positions = positions;
    value = value;
  }

} // namespace
