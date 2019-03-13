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
 */
#ifndef OCLCPP_OCL_TREETENSOR_H_
#define OCLCPP_OCL_TREETENSOR_H_

#include "utils/typedefs.h"
#include "utils/slicing.h"     // Slicable
#include "tensor/functions.h"  // tensor::assign
#include "tensor/column_major.h"  // ColumnMajorVector, assign, subsindex

// This file implements class TreeTensor and static methods on TreeTensor
namespace ocl
{

class TreeTensor : public Slicable
{

 private:
  ColumnMajorVector& value_storage;
  const Tree structure;

 public:

  // Constructor
  TreeTensor(const Tree &structure, const ColumnMajorVector &value_storage)
      : structure(structure), value_storage(value_storage) { }

  // Accessors
  const ColumnMajorVector& value_storage() const { return this->value_storage; }
  const Structure& structure() const { return this->structure; }

  // Return a string representation
  std::string str();

  // Display
  void disp();

  // Sets a value, supports broadcasting
  // tensor::assign itself does broadcasting on the matrix level (dim 1 and 2)
  void set(const Tensor& value)
  {
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      assert(structure.indizes().size()==value.size() || value.size() == 1, "Can not broadcast value.");
      if (structure.indizes().size()==value.size) {
        tensor::assign(indizes[i], value.get(i).data(), value.get(i).size(0), value.get(i).size(1), &value_storage);
      } else {
        // value.size() == 1, broadcast on the third dimension (repeat first matrix)
        tensor::assign(indizes[i], value.get(0), value.get(0).size(0), value.get(0).size(1), &value_storage);
      }
    }
  }

  // Get tensor value of tree tensor
  Tensor value() const
  {
    std::vector<Matrix> matrizes = {};
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      ColumnMajorVector d = tensor::subindex(value_storage, structure.indizes(i));
      ocl::Matrix m(structure.shape(), d);
      matrizes.push_back(m);
    }
    return Tensor(matrizes);
  }

  // Get value data of tree tensor.
  // Returns vector/trajectory of matrizes in column major storage.
  std::vector<ColumnMajorVector> data() const
  {
    std::vector<std::vector<double> > data = {};
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      ColumnMajorVector d = tensor::subindex(value_storage, structure.indizes(i));
      data.push_back(d);
    }
    return data;
  }

  // Return reference to the value storage
  ColumnMajorVector& data() const {
    return value_storage;
  }

  // Returns the size of value
  std::vector<int> size() const { return this->structure().size(); }
  std::vector<int> size(const int dim) const { return this->structure().size(dim); }

  // Returns a sub-tree by id
  TreeTensor get(const std::string& id) const
  {
    Tree r = this->structure().get(id);
    return TreeTensor(r, this->value_storage());
  }

  // Returns a sub-tree by index
  TreeTensor at(const std::vector<int>& indizes) const
  {
    Tree r = this->structure().get(indizes);
    return TreeTensor(r, this->value_storage());
  }

  TreeTensor slice(const std::vector<int>& slice1 = slice::all(this, 0),
                   const std::vector<int>& slice2 = slice::all(this, 1)) const
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
  Tensor plus(const TreeTensor& other) const;
  Tensor minus(const TreeTensor& other) const;
  Tensor ctimes(const TreeTensor& other) const;
  Tensor cdivide(const TreeTensor& other) const;

  Tensor cmin(const TreeTensor& other) const;
  Tensor cmax(const TreeTensor& other) const;

  // binary matrix operations
  Tensor times(const TreeTensor& other) const;
  Tensor cross(const TreeTensor& other) const;
  Tensor dot(const TreeTensor& other) const;

  Tensor atan2(const TreeTensor& other) const;

  // operator overloading
  Tensor operator+(const TreeTensor& other) const;
  Tensor operator-(const TreeTensor& other) const;
  Tensor operator*(const TreeTensor& other) const;
  Tensor operator/(const TreeTensor& other) const;

}; // class StructuredTensor


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

static inline Tensor cpow(const TreeTensor& tt, const TreeTensor& exponent) {
  return ocl::cpow(tt.value(), exponent.value());
}

static inline Tensor norm(const TreeTensor& tt) { return ocl::norm(tt.value()); }
static inline Tensor sum(const TreeTensor& tt) { return ocl::sum(tt.value()); }
static inline Tensor min(const TreeTensor& tt) { return ocl::min(tt.value()); }
static inline Tensor max(const TreeTensor& tt) { return ocl::max(tt.value()); }
static inline Tensor trace(const TreeTensor& tt) { return ocl::trace(tt.value()); }
static inline Tensor mean(const TreeTensor& tt) { return ocl::mean(tt.value()); }

static inline Tensor transpose(const TreeTensor& tt) { return ocl::transpose(tt.value()); }

static inline Tensor reshape(const TreeTensor& tt, const Integer i, const Integer j) {
  return ocl::reshape(tt.value(), i, j);
}

static inline Tensor plus(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::plus(tt1.value(), tt2.value());
}
static inline Tensor minus(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::minus(tt1.value(), tt2.value());
}
static inline Tensor ctimes(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::ctimes(tt1.value(), tt2.value());
}
static inline Tensor cdivide(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::cdivide(tt1.value(), tt2.value());
}

static inline Tensor cmin(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::cmin(tt1.value(), tt2.value());
}
static inline Tensor cmax(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::cmax(tt1.value(), tt2.value());
}

static inline Tensor times(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::times(tt1.value(), tt2.value());
}
static inline Tensor cross(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::cross(tt1.value(), tt2.value());
}
static inline Tensor dot(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::dot(tt1.value(), tt2.value());
}

static inline Tensor atan2(const TreeTensor& tt1, const TreeTensor& tt2) {
  return ocl::atan2(tt1.value(), tt2.value());
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
inline Tensor TreeTensor::cpow(const TreeTensor& exponent) const { return ocl::cpow(*this, exponent); }

// reduction operations
inline Tensor TreeTensor::norm() const { return ocl::norm(*this); }
inline Tensor TreeTensor::sum() const { return ocl::sum(*this); }
inline Tensor TreeTensor::min() const { return ocl::min(*this); }
inline Tensor TreeTensor::max() const { return ocl::max(*this); }
inline Tensor TreeTensor::trace() const { return ocl::trace(*this); }
inline Tensor TreeTensor::mean() const { return ocl::mean(*this); }

// geometrical operations
inline Tensor TreeTensor::transpose() const { return ocl::transpose(*this); }

inline Tensor TreeTensor::reshape(Integer cols, Integer rows) const {
  return ocl::reshape(*this, cols, rows);
}

// get slice (i:j)
inline Tensor TreeTensor::slice(Integer i, Integer j) const {
  return ocl::slice(*this, i, j);
}

// get block slice of cols (i:j) and rows (k:l)
inline Tensor TreeTensor::block(Integer i, Integer j, Integer k, Integer l) const {
  return ocl::block(*this, i, j, k, l);
}

// binary coefficient wise
inline Tensor TreeTensor::plus(const TreeTensor& other) const { return ocl::plus(*this, other); }
inline Tensor TreeTensor::minus(const TreeTensor& other) const { return ocl::minus(*this, other); }
inline Tensor Tensor::ctimes(const TreeTensor& other) const { return ocl::ctimes(*this, other); }
inline Tensor Tensor::cdivide(const TreeTensor& other) const { return ocl::cdivide(*this, other); }

inline Tensor TreeTensor::cmin(const TreeTensor& other) const { return ocl::cmin(*this, other); }
inline Tensor TreeTensor::cmax(const TreeTensor& other) const { return ocl::cmax(*this, other); }

// binary matrix operations
inline Tensor TreeTensor::times(const TreeTensor& other) const { return ocl::times(*this, other); }
inline Tensor TreeTensor::cross(const TreeTensor& other) const { return ocl::cross(*this, other); }
inline Tensor TreeTensor::dot(const TreeTensor& other) const { return ocl::dot(*this, other); }

inline Tensor TreeTensor::atan2(const TreeTensor& other) const { return ocl::atan2(*this, other); }

// operator overloading
inline Tensor TreeTensor::operator+(const TreeTensor& other) const {
  return this->plus(other);
}
inline Tensor TreeTensor::operator-(const TreeTensor& other) const {
  return this->minus(other);
}
inline Tensor TreeTensor::operator*(const TreeTensor& other) const {
  return this->times(other);
}
inline Tensor TreeTensor::operator/(const TreeTensor& other) const {
  return this->cdivide(other);
}

} // namespace ocl
#endif  // OCLCPP_OCL_TREETENSOR_H_
