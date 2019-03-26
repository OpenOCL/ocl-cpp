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
#include <utils/testing.h>
#include "tensor/tensor.h"

TEST(Tensor, aSlice) {

  auto a = ocl::Tensor::Sym(5,5)*10;
  auto r = ocl::slice(a, {2}, ocl::end(a, 1) );

  ocl::test::assertEqual( ocl::full(r,{&a},{ocl::Tensor::One(5,5)}), 10);
}
