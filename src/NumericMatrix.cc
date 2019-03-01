#include "NumericMatrix.h"

inline NumericMatrix uplus(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::uplus(m.m)); }
inline NumericMatrix uminus(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::uminus(m.m)); }
inline NumericMatrix square(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::square(m.m)); }
inline NumericMatrix inverse(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::inverse(m.m)); }
inline NumericMatrix abs(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::abs(m.m)); }
inline NumericMatrix sqrt(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sqrt(m.m)); }
inline NumericMatrix sin(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sin(m.m)); }
inline NumericMatrix cos(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::cos(m.m)); }
inline NumericMatrix tan(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::tan(m.m)); }
inline NumericMatrix atan(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::atan(m.m)); }
inline NumericMatrix asin(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::asin(m.m)); }
inline NumericMatrix acos(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::acos(m.m)); }
inline NumericMatrix tanh(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::tanh(m.m)); }
inline NumericMatrix sinh(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sinh(m.m)); }
inline NumericMatrix cosh(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::cosh(m.m)); }
inline NumericMatrix exp(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::exp(m.m)); }
inline NumericMatrix log(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::log(m.m)); }

inline NumericMatrix pow(const NumericMatrix& m, const Scalar exponent) { return NumericMatrix(ocl::eigen::pow(m.m, exponent)); }

inline NumericMatrix norm(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::norm(m.m)); }
inline NumericMatrix sum(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sum(m.m)); }
inline NumericMatrix min(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::min(m.m)); }
inline NumericMatrix max(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::max(m.m)); }
inline NumericMatrix mean(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::mean(m.m)); }
inline NumericMatrix trace(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::trace(m.m)); }
inline NumericMatrix prod(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::prod(m.m)); }

inline NumericMatrix reshape(const NumericMatrix& m, const Integer rows, const Integer cols) {
  return NumericMatrix(ocl::eigen::reshape(m.m, rows, cols));
}
inline NumericMatrix transpose(const NumericMatrix& m) {
  return NumericMatrix(ocl::eigen::transpose(m.m));
}
inline NumericMatrix block(const NumericMatrix& m, const Integer i, const Integer j, const Integer k, const Integer l) {
  return NumericMatrix(ocl::eigen::block(m.m, i, j, k, l));
}
inline NumericMatrix slice(const NumericMatrix& m, const Integer i, const Integer k) {
  return NumericMatrix(ocl::eigen::slice(m.m, i, k));
}

inline NumericMatrix ctimes(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(ocl::eigen::ctimes(m1.m, m2.m)); }
inline NumericMatrix cplus(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(ocl::eigen::cplus(m1.m, m2.m)); }
inline NumericMatrix cdiv(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(ocl::eigen::cdiv(m1.m, m2.m)); }
inline NumericMatrix cminus(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::cminus(m1.m, m2.m)); }

inline NumericMatrix times(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::times(m1.m, m2.m)); }
inline NumericMatrix cross(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::cross(m1.m, m2.m)); }
inline NumericMatrix dot(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::dot(m1.m, m2.m)); }
