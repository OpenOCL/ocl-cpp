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
#ifndef OCL_CASADI_H_
#define OCL_CASADI_H_

#include "casadi/casadi.hpp"

namespace ocl
{
typedef ::casadi::SX CasadiMatrix; // native casadi type
typedef ::casadi::SX CasadiScalar;
typedef int CasadiInteger;

namespace casadi
{

static inline CasadiMatrix Sym(int rows, int cols) {
  return CasadiMatrix::sym("m", rows, cols);
}

static inline CasadiMatrix Eye(int n) {
  return CasadiMatrix::eye(n);
}

static inline CasadiMatrix Zero(int rows, int cols) {
  return CasadiMatrix::zeros(rows, cols);
}

static inline CasadiMatrix One(int rows, int cols) {
  return CasadiMatrix::ones(rows, cols);
}

static inline void assign(CasadiMatrix& m, const int row, const int col, const double value)
{
  // false means zero based indexing (true is one based like in Matlab)
  m.set(value, false, row, col);
}

static inline void assign(CasadiMatrix& m, const std::vector<int>& rows,
                          const int col, const CasadiMatrix& values)
{
  // false means zero based indexing (true is one based like in Matlab)
  m.set(values, false, rows, col);
}

static inline std::vector<int> shape(const CasadiMatrix& m)
{
  return {(int)m.rows(), (int)m.columns()};
}

static inline int size(const CasadiMatrix& m, const int dim)
{
  return m.size(dim+1); // one based indexing in casadi
}

static inline std::vector<double> full(const CasadiMatrix& m)
{
  std::string name = "f";
  std::vector<CasadiMatrix> f_inputs;
  std::vector<CasadiMatrix> f_outputs;
  f_outputs.push_back(m);

  ::casadi::Dict opts = ::casadi::Dict();

  ::casadi::Function f = ::casadi::Function(name, f_inputs, f_outputs, opts);
  std::vector< ::casadi::DM > dm_out;
  f.call({},dm_out);

  ::casadi::DM d = dm_out[0];

  double *data = d.ptr();
  int nel = size(m,0)*size(m,1);
  std::vector<double> values(data, data + nel);
  return values;
}

// native casadi type operations
static inline CasadiMatrix uplus(const CasadiMatrix& m) { return m; }
static inline CasadiMatrix uminus(const CasadiMatrix& m) { return -m; }
static inline CasadiMatrix square(const CasadiMatrix& m) { return CasadiMatrix::sq(m); }
static inline CasadiMatrix inverse(const CasadiMatrix& m) { return CasadiMatrix::inv(m); }
static inline CasadiMatrix abs(const CasadiMatrix& m) { return CasadiMatrix::abs(m); }
static inline CasadiMatrix sqrt(const CasadiMatrix& m) { return CasadiMatrix::sqrt(m); }
static inline CasadiMatrix sin(const CasadiMatrix& m) { return CasadiMatrix::sin(m); }
static inline CasadiMatrix cos(const CasadiMatrix& m) { return CasadiMatrix::cos(m); }
static inline CasadiMatrix tan(const CasadiMatrix& m) { return CasadiMatrix::tan(m); }
static inline CasadiMatrix atan(const CasadiMatrix& m) { return CasadiMatrix::atan(m); }
static inline CasadiMatrix asin(const CasadiMatrix& m) { return CasadiMatrix::asin(m); }
static inline CasadiMatrix acos(const CasadiMatrix& m) { return CasadiMatrix::acos(m); }
static inline CasadiMatrix tanh(const CasadiMatrix& m) { return CasadiMatrix::tanh(m); }
static inline CasadiMatrix sinh(const CasadiMatrix& m) { return CasadiMatrix::sinh(m); }
static inline CasadiMatrix cosh(const CasadiMatrix& m) { return CasadiMatrix::cosh(m); }
static inline CasadiMatrix exp(const CasadiMatrix& m) { return CasadiMatrix::exp(m); }
static inline CasadiMatrix log(const CasadiMatrix& m) { return CasadiMatrix::log(m); }

static inline CasadiMatrix cpow(const CasadiMatrix& m, const CasadiMatrix& exponent) {
  return CasadiMatrix::pow(m, exponent);
}

// reduction
static inline CasadiMatrix norm(const CasadiMatrix& m) {
  return CasadiMatrix::norm_2(m);
}
static inline CasadiMatrix sum(const CasadiMatrix& m) { return CasadiMatrix::sum1(CasadiMatrix::sum2(m)); }
static inline CasadiMatrix min(const CasadiMatrix& m) { return CasadiMatrix::mmin(m); }
static inline CasadiMatrix max(const CasadiMatrix& m) { return CasadiMatrix::mmax(m); }
static inline CasadiMatrix mean(const CasadiMatrix& m) { return sum(m)/(m.rows()*m.columns()); }
static inline CasadiMatrix trace(const CasadiMatrix& m) { return CasadiMatrix::trace(m); }

// geometrical
static inline CasadiMatrix reshape(const CasadiMatrix& m, CasadiInteger rows, CasadiInteger cols) {
  return CasadiMatrix::reshape(m, rows, cols);
}
static inline CasadiMatrix transpose(const CasadiMatrix& m) { return m.T(); }

static inline CasadiMatrix slice(const CasadiMatrix& m, const std::vector<int>& slice1, const std::vector<int>& slice2) {
  CasadiMatrix ret = m(slice1, slice2);
  return ret;
}

// binary coefficient wise
static inline CasadiMatrix ctimes(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::times(m1, m2);
}
static inline CasadiMatrix plus(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return m1 + m2;
}
static inline CasadiMatrix cdivide(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::mrdivide(m1,m2);
}
static inline CasadiMatrix minus(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return m1 - m2;
}

static inline CasadiMatrix cmin(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::fmin(m1, m2);
}

static inline CasadiMatrix cmax(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::fmax(m1, m2);
}

// binary operations
static inline CasadiMatrix times(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::mtimes(m1,m2);
}

static inline CasadiMatrix cross(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::cross(m1, m2);
}

static inline CasadiMatrix dot(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::dot(m1, m2);
}

static inline CasadiMatrix atan2(const CasadiMatrix& m1, const CasadiMatrix& m2) {
  return CasadiMatrix::atan2(m1, m2);
}

}
} // namespace ocl
#endif // OCL_CASADI_H_
