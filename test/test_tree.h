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
#include "tensor/tree_builder.h"

TEST(Tree, bTwoVariables)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});

  ocl::Tree x = tb.tree();

  std::vector<std::vector<int>> a = {{0,1}};
  ocl::test::assertEqual(x.get("x1").indizes(), a, OCL_INFO);
  ocl::test::assertEqual(x.get("x2").indizes(),{{2,3,4,5,6,7}}, OCL_INFO);
  ocl::test::assertEqual(x.shape(),{8,1}, OCL_INFO);
}

TEST(Tree, cShape)
{
  ocl::TreeBuilder tb;
  tb.add("x1",{1,8});

  ocl::Tree x = tb.tree();

  ocl::test::assertEqual(x.shape(), {8,1});
}

TEST(Tree, dRepeatedVar) {
  ocl::TreeBuilder tb;
  tb.add("x1",{1,3});
  tb.add("x2",{3,2});
  tb.add("x1",{1,3});

  ocl::Tree x = tb.tree();

  ocl::test::assertEqual(x.get("x1").indizes(),{{0,1,2},{9,10,11}}, OCL_INFO);
  ocl::test::assertEqual(x.get("x2").indizes(),{{3,4,5,6,7,8}}, OCL_INFO);
}

TEST(Tree, eSubTree)
{
  ocl::TreeBuilder tb_u;
  tb_u.add("x1",{1,3});
  tb_u.add("x3",{3,3});
  tb_u.add("x1",{1,3});
  tb_u.add("x3",{3,3});

  ocl::Tree u = tb_u.tree();

  ocl::TreeBuilder tb_x;
  tb_x.add("x1",{1,3});
  tb_x.add("u",u);
  tb_x.add("u",u);
  tb_x.add("x2",{3,2});
  tb_x.add("x1",{1,3});

  ocl::Tree x = tb_x.tree();

  ocl::test::assertEqual(x.get("u").get("x1").indizes(), {{3,4,5},{15,16,17},{27,28,29},{39,40,41}}, OCL_INFO);
  //
  // ocl::Tree r = x.get("u").at(0).get("x1");
  // ocl::test::assertEqual(r.indizes(), {{3,4,5},{15,16,17}} );
}

TEST(Tree, fSliceMatrix)
{

  ocl::Tree m = ocl::Leaf({4,4});

  ocl::test::assertEqual(m.indizes(), {ocl::linspace(0,15)}, OCL_INFO);

  ocl::Tree n = m.slice({0,1},{1,2});

  ocl::test::assertEqual(n.indizes(), {{4,5,8,9}}, OCL_INFO);
}
