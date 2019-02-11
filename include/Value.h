#ifndef OCPCPP_OCL_NUMERICVALUE_H_
#define OCPCPP_OCL_NUMERICVALUE_H_

#include "typedefs.h"

namespace ocl 
{
 
template<class T>
class Value
{

 private:
  const T &value;
  const Structure &structure;
  const PositionArray &positions;

 public:

  // Static factory methods
  static Value<T> create(const Structure &structure, const T &value);
  static Value<T> create(const Structure &structure, const float[] &value);
  static Value<T> createFromValue(const Structure &structure, 
      const PositionArray &positions, const Value<T> &value);
  static Value<T> Matrix(const float[] &value);

  // Constructor
  Value(const Structure &structure, const PositionArray &positions, const T &value);
  // Returns the number of elements of the value
  int numel();
  // Returns the underlying value
  T value();
  // Return a string representation
  std::string str();
  // Display
  void disp();
  // Sets a value, supports broadcasting, optional slices
  void set(T value, std::string slice1=":",
      std::string slice2=":", std::string slice3=":");
  // Returns the size of value
  Size size();
  // Returns a sub-value by id
  T get(std::string id);
  // Slices value
  T slice(std::string slice1=":",
      std::string slice2=":", std::string slice3=":");

  // linspace operator
  T = linspace(const Value<T>& other);

  // operators - unary
  T uplus() return Value::Matrix(value);
  T uminus() return Value::Matrix(-value);
  T ctranspose() return this->transpose();
  T transpose() return Value::Matrix(value->transpose());
  T triu();
  T sum();
  T norm();
  T det();
  T trace();
  T diag();
  T abs();
  T sqrt();
  T sin();
  T cos();
  T tan();
  T atan();
  T asin();
  T acos();
  T tanh();
  T cosh();
  T sinh();
  T acosh();
  T exp();
  T log();
  // operators - unary + constant
  T reshape(int[]);
  T repmat(int[]);


} // namespace ocl 


#endif  // OCPCPP_OCL_NUMERICVALUE_H_