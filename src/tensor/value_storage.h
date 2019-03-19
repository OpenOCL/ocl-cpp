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
#ifndef OCL_VALUE_STORAGE_H_
#define OCL_VALUE_STORAGE_H_

namespace ocl {

// Stores matrix data in column major format
class ValueStorage : public Slicable
{
public:

  // Reshape matrizes to vectors
  ValueStorage(const CasadiMatrix& m)
      : m( casadi::reshape(m, casadi::size(m, 0)*casadi::size(m, 1), 1) ) { }

  ValueStorage(const int size)
      : m(casadi::Zero(size, 1)) { }

  ValueStorage(const int size, const double val)
      : m(casadi::One(size, 1) * val) { }

  // ValueStorage(const int size, const std::vector<double>& values) {
  //   this->assign(all(*this, 0), values, size, 1);
  // }

  virtual int size(const int dim) const override {
    return m.size(dim);
  }

  std::vector<double> data() const {
    return casadi::full(m);
  }

  ValueStorage subsindex(const std::vector<int>& indizes) const {
    return ValueStorage(casadi::slice(m, indizes, {0}));
  }

  void assign(const std::vector<int>& indizes, const CasadiMatrix& values, int size0, int size1) {
    casadi::assign(m, indizes, 0, values);
  }

  void assign(const std::vector<int>& indizes, const std::vector<double>& values, int size0, int size1)
  {
    CasadiMatrix values_m(values);
    assign(indizes, values_m, size0, size1);
  }

private:
  CasadiMatrix m;
};

} // namespace ocl
#endif // OCL_VALUE_STORAGE_H_
