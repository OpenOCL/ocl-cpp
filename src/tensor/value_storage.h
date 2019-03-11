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
#ifndef OCLCPP_OCL_VALUESTORAGE_H_
#define OCLCPP_OCL_VALUESTORAGE_H_

namespace ocl
{

class ValueStorage
{
 public:

  ValueStorage();

  // number of elements, size of storage
  int nel() { storage.size(); }

  // set trajectory of value
  void set(const std::vector<int>& indizes, const std::vector<Tensor>& value)
  {
    assertEqual(indizes.size(), value.size());

    for(unsigned int i=0; i < indizes.size(); i++)
    {
      storage.set(indizes[i]) = value[i];
    }
  }

  // set value
  void set(const std::vector<int>& indizes, const Tensor& value)
  {
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      storage.set(indizes[i]) = value;
    }
  }

  Tensor value(const std::vector<int>& indizes, const Shape& shape)
  {
    std::vector<T> vout = {};
    for(unsigned int i=0; i < indizes.size(); i++)
    {
      T v = storage.slice(structure.indizes[i]);
      v = v.reshape(shape);
      vout[i] = v;
    }
    return Tensor(vout);
  }

 private:
  Tensor storage;
};

} // namespace ocl
#endif  // OCLCPP_OCL_VALUESTORAGE_H_
