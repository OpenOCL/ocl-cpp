#include <utils/testing.h>
#include "tensor/tree_builder.h"
#include "tensor/tree_tensor.h"

TEST(TreeTensor, aThreeVariablesSet)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree x_structure = tb.tree();

  ocl::ValueStorage vs(x_structure.numel(), 4);
  ocl::TreeTensor x(x_structure, vs);

  x.set(ocl::Matrix({1,2,3,4,5,6,7,8,9,10}));
  ocl::test::assertEqual(x.data(), {{1,2,3,4,5,6,7,8,9,10}}, OCL_INFO);
}

TEST(TreeTensor, bThreeVariablesAllSlice)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree x_structure = tb.tree();

  ocl::ValueStorage vs(x_structure.numel(), 4);
  ocl::TreeTensor x(x_structure, vs);

  x.slice(ocl::all(x, 0), {0}).set(ocl::Matrix({1,2,3,4,5,6,7,8,9,10}));

  ocl::test::assertEqual(x.data(), {{1,2,3,4,5,6,7,8,9,10}}, OCL_INFO);
  ocl::test::assertEqual(vs.data(), {1,2,3,4,5,6,7,8,9,10}, OCL_INFO);
}

TEST(TreeTensor, cThreeVariablesSubsrefSlice)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree x_structure = tb.tree();

  ocl::ValueStorage vs(x_structure.numel(), 4);
  ocl::TreeTensor x(x_structure, vs);

  x.set(ocl::Matrix({1,2,3,4,5,6,7,8,9,10}));
  ocl::test::assertEqual(x.get("x1").data(), {{1,2},{9,10}}, OCL_INFO);

  ocl::test::assertEqual(x.get("x1").slice({0},{0}).data(), {{1},{9}}, OCL_INFO);
}

TEST(TreeTensor, dStateTensor)
{
  ocl::TreeBuilder tb_x;
  tb_x.add("p",{3,1});
  tb_x.add("R",{3,3});
  tb_x.add("v",{3,1});
  tb_x.add("w",{3,1});

  ocl::Tree x_structure = tb_x.tree();
  ocl::ValueStorage vs(x_structure.numel(), 0);
  ocl::TreeTensor state(x_structure, vs);

  state.get("p").set(ocl::Matrix({100, 0, -50}));
  state.get("R").set(ocl::Matrix::Eye(3));
  state.get("v").set(ocl::Matrix({20, 0, 0}));
  state.get("w").set(ocl::Matrix({0, 1, 0.1}));

  ocl::test::assertEqual( state.get("p").data(),   {{100, 0, -50}}, OCL_INFO);
  ocl::test::assertEqual( state.get("R").data(),   {ocl::Matrix::Eye(3).full()}, OCL_INFO);
  ocl::test::assertEqual( state.get("v").data(),   {{20, 0, 0}}, OCL_INFO);
  ocl::test::assertEqual( state.get("w").data(),   {{0, 1, 0.1}}, OCL_INFO);
}

TEST(TreeTensor, eOcpTensor)
{
  ocl::TreeBuilder tb_x;
  tb_x.add("p",{3,1});
  tb_x.add("R",{3,3});
  tb_x.add("v",{3,1});
  tb_x.add("w",{3,1});

  ocl::TreeBuilder tb_u;
  tb_u.add("elev",{1,1});
  tb_u.add("ail",{1,1});

  ocl::TreeBuilder tb_ocp;
  tb_ocp.addRepeated({"x","u"},{tb_x.tree(), tb_u.tree()}, 5);
  tb_ocp.add("x", tb_x.tree());

  ocl::Tree ocp_tree = tb_ocp.tree();

  ocl::ValueStorage vs(ocp_tree.numel(), 0);
  ocl::TreeTensor v(ocp_tree, vs);

  ocl::TreeTensor x = v.get("x");
  ocl::TreeTensor p = x.get("p");
  p.set(ocl::Matrix({100, 0, -50}));

  v.get("x").get("p").set(ocl::Matrix({100, 0, -50}));
  v.get("x").get("R").set(ocl::Matrix::Eye(3));
  v.get("x").get("v").set(ocl::Matrix({20, 0, 0}));
  v.get("x").get("w").set(ocl::Matrix({0, 1, 0.1}));

  std::vector<int> all = ocl::all(v.get("x"), 1);
  ocl::TreeTensor sliced = v.get("x").slice({0}, all);
  ocl::TreeTensor subsindexed = sliced.at(ocl::linspace(3,5));
  ocl::test::assertEqual( subsindexed.data(), {{100}, {100}, {100}}, OCL_INFO);

  v.get("x").get("R").set(ocl::Matrix::Eye(3));

  ocl::test::assertEqual(v.get("x").get("R").data(), {{1,0,0,0,1,0,0,0,1},{1,0,0,0,1,0,0,0,1},{1,0,0,0,1,0,0,0,1},{1,0,0,0,1,0,0,0,1},{1,0,0,0,1,0,0,0,1},{1,0,0,0,1,0,0,0,1}}, OCL_INFO);
  ocl::test::assertEqual(v.get("x").get("R").at({1}).data(), {ocl::Matrix::Eye(3).full()}, OCL_INFO);

  v.get("x").get("R").set(ocl::Matrix::One(3,3));
  ocl::test::assertEqual(v.get("x").get("R").at({0,1}).data(), {{1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1}}, OCL_INFO);
  ocl::test::assertEqual(v.get("x").get("p").at({1}).data(), {{100, 0, -50}}, OCL_INFO);

  v.get("x").at(ocl::end(v.get("x"), 2)).set({ocl::linspace(2., 19.)});
  ocl::test::assertEqual(v.get("x").at(ocl::end(v.get("x"), 2)).slice({1},{0}).data(), {{3.}}, OCL_INFO);

  ocl::TreeTensor xend = v.get("x").at( ocl::end(v.get("x"), 2) );
  ocl::test::assertEqual( xend.slice( ocl::end(v.get("x"), 0), {0}).data(), {{19}}, OCL_INFO);
  ocl::test::assertEqual( xend.slice({1}, {0}).data(), {{3.}}, OCL_INFO);
}
