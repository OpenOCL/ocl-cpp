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
 *    You should have received a copy of the GNU General Public
 *    License along with this program;
 *    if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */
#ifndef OCLCPP_OCL_VALUESTORAGE_H_
#define OCLCPP_OCL_VALUESTORAGE_H_

namespace ocl
{

template <class T>;
class ValueStorage
{
 public:

  ValueStorage();

  // number of elements, size of storage
  int nel() { storage.size(); }

  // set trajectory of value
  void set(const RootNode& structure, const std::vector<T>& value)
  {
    assertEqual(structure.length(), value.size());

    for(int i=0; i < structure.length(); i++)
    {
      std::vector<int> idz = structure.indizes[i];
      T v = value[i];
      storage.set(idz) = v;
    }
  }

  // set value
  void set(const RootNode& structure, const T& value)
  {
    for(int i=0; i < structure.length(); i++)
    {
      std::vector<int> idz = structure.indizes[i];
      storage.set(idz) = value;
    }
  }

  std::vector<T> value(const RootNode$ structure)
  {
    Shape s = structure.shape();

    std::vector<T> vout = {};
    for(int i=0; i < structure.length(); i++)
    {
      T v = storage[structure.indizes[i]];
      v = v.reshape(s);
      vout[i] = v;
    }
  }

 private:
  T storage;
};

} // namespace ocl
#endif  // OCLCPP_OCL_VALUESTORAGE_H_