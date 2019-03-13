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
#ifndef OCL_COLUMN_MAJOR_H_
#define OCL_COLUMN_MAJOR_H_


namespace ocl {

static inline void assign(std::vector<int> indizes, std::vector<double> values, int dim0, int dim1, ColumnMajorVector *value_storage)
{

}

static inline std::vector<double> subsindex(const ColumnMajorVector& values, const std::vector<int>& indizes)
{

  return v_out;
}

// Stores matrix data in column major format
class ColumnMajorVector
{
public:
  ColumnMajorVector(int size) : m(Matrix::Zero(size,1)) { }

private:
  Matrix m;
};

} // namespace ocl
#endif // OCL_COLUMN_MAJOR_H_
