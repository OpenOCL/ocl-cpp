/*
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
#ifndef OCL_TREETENSOR_H_
#define OCL_TREETENSOR_H_

#include "utils/typedefs.h"
#include "utils/assertions.h"      // assertTrue
#include "utils/slicing.h"         // Slicable
#include "tensor/value_storage.h"  // ValueStorage, assign, subsindex

// This file implements class TreeTensor and static functions on TreeTensor
namespace ocl
{

class TreeTensor : public Slicable
{

 public:

  // Constructor
  TreeTensor(const Tree& structure, ValueStorage& value_storage)
      : _structure(structure), _value_storage(value_storage) { }

  // Accessors
  ValueStorage& value_storage() const { return this->_value_storage; }
  Tree structure() const { return this->_structure; }

  // Return a string representation
  std::string str();

  // Display
  void disp();

  // Sets a value, supports broadcasting
  // TODO: ValueStorage::assign itself should support broadcasting on the matrix level (dim 1 and 2)
  void set(const Tensor& value)
  {
    std::vector<std::vector<int> > indizes = this->structure().indizes();
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      if ((int)indizes.size()==value.size()) {
        this->_value_storage.assign(indizes[i], value.get(i).data());
      }
      else if (value.size() == 1) {
        // broadcast on the third dimension (repeat first matrix)
        this->_value_storage.assign(indizes[i], value.get(0).data());
      }
      else {
        assertEqual(0,1,"wrong");
      }
    }
  }

  // Get tensor value of tree tensor
  Tensor value() const
  {
    std::vector<std::vector<int> > indizes = this->structure().indizes();
    std::vector<Matrix> matrizes = {};
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      ValueStorage vs = this->value_storage().subsindex(this->structure().indizes(i));

      Matrix m = vs.data();
      std::vector<int> s = this->structure().shape();
      m = m.reshape(s[0], s[1]);
      matrizes.push_back(m);
    }
    return Tensor(matrizes);
  }

  // Get value data of tree tensor.
  // Returns vector/trajectory of matrizes in column major storage.
  std::vector<std::vector<double> > data() const
  {
    std::vector<std::vector<int> > indizes = this->structure().indizes();
    std::vector<std::vector<double> > data;
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      auto idz = this->structure().indizes(i);
      ValueStorage vs = this->value_storage().subsindex(idz);
      data.push_back(vs.data());
    }
    return data;
  }

  // Returns the size of value
  int size() const { return this->structure().size(); }
  virtual int size(const int dim) const override { return this->structure().size(dim); }

  // Returns a sub-tree by id
  TreeTensor get(const std::string& id) const
  {
    Tree r = this->structure().get(id);
    return TreeTensor(r, this->_value_storage);
  }

  // Returns a sub-tree by index
  TreeTensor at(const std::vector<int>& indizes) const
  {
    Tree r = this->structure().at(indizes);
    return TreeTensor(r, this->value_storage());
  }

  TreeTensor slice(const std::vector<int>& slice1,
                   const std::vector<int>& slice2) const
  {
    Tree r = this->structure().slice(slice1, slice2);
    return TreeTensor(r, this->value_storage());
  }

  // operators - unary element wise
  Tensor uplus() const;
  Tensor uminus() const;
  Tensor square() const;
  Tensor inverse() const;
  Tensor abs() const;
  Tensor sqrt() const;
  Tensor sin() const;
  Tensor cos() const;
  Tensor tan() const;
  Tensor atan() const;
  Tensor asin() const;
  Tensor acos() const;
  Tensor tanh() const;
  Tensor cosh() const;
  Tensor sinh() const;
  Tensor exp() const;
  Tensor log() const;

  // operators - unary element wise + scalar
  Tensor cpow(const Tensor& exponent) const;

  // reduction operations
  Tensor norm() const;
  Tensor sum() const;
  Tensor min() const;
  Tensor max() const;
  Tensor trace() const;
  Tensor mean() const;

  // geometrical operations
  Tensor transpose() const;

  Tensor reshape(int cols, int rows) const;

  // binary coefficient wise
  Tensor plus(const Tensor& other) const;
  Tensor minus(const Tensor& other) const;
  Tensor ctimes(const Tensor& other) const;
  Tensor cdivide(const Tensor& other) const;

  Tensor cmin(const Tensor& other) const;
  Tensor cmax(const Tensor& other) const;

  // binary matrix operations
  Tensor times(const Tensor& other) const;
  Tensor cross(const Tensor& other) const;
  Tensor dot(const Tensor& other) const;

  Tensor atan2(const Tensor& other) const;

  // operator overloading
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

private:
 Tree _structure;
 ValueStorage& _value_storage;

}; // class TreeTensor


static inline Tensor uplus(const TreeTensor& tt) { return ocl::uplus(tt.value()); }
static inline Tensor uminus(const TreeTensor& tt) { return ocl::uminus(tt.value()); }
static inline Tensor square(const TreeTensor& tt) { return ocl::square(tt.value()); }
static inline Tensor inverse(const TreeTensor& tt) { return ocl::inverse(tt.value()); }
static inline Tensor abs(const TreeTensor& tt) { return ocl::abs(tt.value()); }
static inline Tensor sqrt(const TreeTensor& tt) { return ocl::sqrt(tt.value()); }
static inline Tensor sin(const TreeTensor& tt) { return ocl::sin(tt.value()); }
static inline Tensor cos(const TreeTensor& tt) { return ocl::cos(tt.value()); }
static inline Tensor tan(const TreeTensor& tt) { return ocl::tan(tt.value()); }
static inline Tensor atan(const TreeTensor& tt) { return ocl::atan(tt.value()); }
static inline Tensor asin(const TreeTensor& tt) { return ocl::asin(tt.value()); }
static inline Tensor acos(const TreeTensor& tt) { return ocl::acos(tt.value()); }
static inline Tensor tanh(const TreeTensor& tt) { return ocl::tanh(tt.value()); }
static inline Tensor cosh(const TreeTensor& tt) { return ocl::cosh(tt.value()); }
static inline Tensor sinh(const TreeTensor& tt) { return ocl::sinh(tt.value()); }
static inline Tensor exp(const TreeTensor& tt) { return ocl::exp(tt.value()); }
static inline Tensor log(const TreeTensor& tt) { return ocl::log(tt.value()); }

static inline Tensor cpow(const TreeTensor& tt, const Tensor& exponent) {
  return ocl::cpow(tt.value(), exponent);
}

static inline Tensor norm(const TreeTensor& tt) { return ocl::norm(tt.value()); }
static inline Tensor sum(const TreeTensor& tt) { return ocl::sum(tt.value()); }
static inline Tensor min(const TreeTensor& tt) { return ocl::min(tt.value()); }
static inline Tensor max(const TreeTensor& tt) { return ocl::max(tt.value()); }
static inline Tensor trace(const TreeTensor& tt) { return ocl::trace(tt.value()); }
static inline Tensor mean(const TreeTensor& tt) { return ocl::mean(tt.value()); }

static inline Tensor transpose(const TreeTensor& tt) { return ocl::transpose(tt.value()); }

static inline Tensor reshape(const TreeTensor& tt, const int i, const int j) {
  return ocl::reshape(tt.value(), i, j);
}

static inline Tensor plus(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::plus(tt1.value(), tt2);
}
static inline Tensor minus(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::minus(tt1.value(), tt2);
}
static inline Tensor ctimes(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::ctimes(tt1.value(), tt2);
}
static inline Tensor cdivide(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::cdivide(tt1.value(), tt2);
}

static inline Tensor cmin(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::cmin(tt1.value(), tt2);
}
static inline Tensor cmax(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::cmax(tt1.value(), tt2);
}

static inline Tensor times(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::times(tt1.value(), tt2);
}
static inline Tensor cross(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::cross(tt1.value(), tt2);
}
static inline Tensor dot(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::dot(tt1.value(), tt2);
}

static inline Tensor atan2(const TreeTensor& tt1, const Tensor& tt2) {
  return ocl::atan2(tt1.value(), tt2);
}

//
// Define TreeTensor operations

// operators - unary element wise
inline Tensor TreeTensor::uplus() const { return ocl::uplus(*this); }
inline Tensor TreeTensor::uminus() const { return ocl::uminus(*this); }
inline Tensor TreeTensor::square() const { return ocl::square(*this); }
inline Tensor TreeTensor::inverse() const { return ocl::inverse(*this); }
inline Tensor TreeTensor::abs() const { return ocl::abs(*this); }
inline Tensor TreeTensor::sqrt() const { return ocl::sqrt(*this); }
inline Tensor TreeTensor::sin() const { return ocl::sin(*this); }
inline Tensor TreeTensor::cos() const { return ocl::cos(*this); }
inline Tensor TreeTensor::tan() const { return ocl::tan(*this); }
inline Tensor TreeTensor::atan() const { return ocl::atan(*this); }
inline Tensor TreeTensor::asin() const { return ocl::asin(*this); }
inline Tensor TreeTensor::acos() const { return ocl::acos(*this); }
inline Tensor TreeTensor::tanh() const { return ocl::tanh(*this); }
inline Tensor TreeTensor::cosh() const { return ocl::cosh(*this); }
inline Tensor TreeTensor::sinh() const { return ocl::sinh(*this); }
inline Tensor TreeTensor::exp() const { return ocl::exp(*this); }
inline Tensor TreeTensor::log() const { return ocl::log(*this); }

// operators - unary element wise + scalar
inline Tensor TreeTensor::cpow(const Tensor& exponent) const { return ocl::cpow(*this, exponent); }

// reduction operations
inline Tensor TreeTensor::norm() const { return ocl::norm(*this); }
inline Tensor TreeTensor::sum() const { return ocl::sum(*this); }
inline Tensor TreeTensor::min() const { return ocl::min(*this); }
inline Tensor TreeTensor::max() const { return ocl::max(*this); }
inline Tensor TreeTensor::trace() const { return ocl::trace(*this); }
inline Tensor TreeTensor::mean() const { return ocl::mean(*this); }

// geometrical operations
inline Tensor TreeTensor::transpose() const { return ocl::transpose(*this); }

inline Tensor TreeTensor::reshape(const int cols, const int rows) const {
  return ocl::reshape(*this, cols, rows);
}

// binary coefficient wise
inline Tensor TreeTensor::plus(const Tensor& other) const { return ocl::plus(*this, other); }
inline Tensor TreeTensor::minus(const Tensor& other) const { return ocl::minus(*this, other); }
inline Tensor TreeTensor::ctimes(const Tensor& other) const { return ocl::ctimes(*this, other); }
inline Tensor TreeTensor::cdivide(const Tensor& other) const { return ocl::cdivide(*this, other); }

inline Tensor TreeTensor::cmin(const Tensor& other) const { return ocl::cmin(*this, other); }
inline Tensor TreeTensor::cmax(const Tensor& other) const { return ocl::cmax(*this, other); }

// binary matrix operations
inline Tensor TreeTensor::times(const Tensor& other) const { return ocl::times(*this, other); }
inline Tensor TreeTensor::cross(const Tensor& other) const { return ocl::cross(*this, other); }
inline Tensor TreeTensor::dot(const Tensor& other) const { return ocl::dot(*this, other); }

inline Tensor TreeTensor::atan2(const Tensor& other) const { return ocl::atan2(*this, other); }

// operator overloading
inline Tensor TreeTensor::operator+(const Tensor& other) const {
  return this->plus(other);
}
inline Tensor TreeTensor::operator-(const Tensor& other) const {
  return this->minus(other);
}
inline Tensor TreeTensor::operator*(const Tensor& other) const {
  return this->times(other);
}
inline Tensor TreeTensor::operator/(const Tensor& other) const {
  return this->cdivide(other);
}

} // namespace ocl
#endif  // OCL_TREETENSOR_H_
