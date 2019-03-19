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
class ValueStorage
{
public:

  // Reshape matrizes to vectors
  ValueStorage(const CasadiMatrix& m) : m(m.reshape(m.size(0)*m.size(1), 1)) { }
  ValueStorage(int size) : m(Matrix::Zero(size,1)) { }

  Matrix data() const { return m; }

  ValueStorage subsindex(const std::vector<int>& indizes) const
  {
    return ColumnMajorVector(m.slice(indizes, 0));
  }

  void assign(const std::vector<int>& indizes, const ValueStorage& values, int size0, int size1)
  {
    m.assign(indizes, 0, values.data());
  }

private:
  CasadiMatrix m;
};

} // namespace ocl
#endif // OCL_VALUE_STORAGE_H_
