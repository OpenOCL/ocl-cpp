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
#include "Value.h"

namespace ocl {

static StructuredTensor StructuredTensor::create(const Structure &structure,
    const Tensor &tensor)
{
  s = structure->size();
  PositionArray positions(s);
  positions = value;
  oclValue = StructuredTensor(structure, value);
}

// constructor
StructuredTensor(const Structure &structure, const PositionArray &positions, const T &value)
{
  structure = structure;
  positions = positions;
  value = value;
}

} // namespace
