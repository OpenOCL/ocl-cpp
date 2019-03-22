#include <utils/testing.h>
#include "tensor/tree_builder.h"
#include "tensor/tree_tensor.h"

TEST(TreeTensor, aThreeVariablesSet)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree& x_structure = tb.tree();

  ocl::ValueStorage vs(x_structure.numel(), 4);
  ocl::TreeTensor x(x_structure, vs);

  x.set(ocl::Matrix({1,2,3,4,5,6,7,8,9,10}));
  ocl::test::assertEqual(x.data(), {{1,2,3,4,5,6,7,8,9,10}});
}

TEST(TreeTensor, bThreeVariablesAllSlice)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree& x_structure = tb.tree();

  ocl::ValueStorage vs(x_structure.numel(), 4);
  ocl::TreeTensor x(x_structure, vs);

  // x.slice(ocl::all(x, 0), {0}).set(ocl::Matrix({1,2,3,4,5,6,7,8,9,10}));

  // ocl::test::assertEqual(x.data(), {{1,2,3,4,5,6,7,8,9,10}});
  // ocl::test::assertEqual(vs.data(), {1,2,3,4,5,6,7,8,9,10});
}

TEST(TreeTensor, cThreeVariablesSubsrefSlice)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree& x_structure = tb.tree();

  ocl::ValueStorage vs(x_structure.numel(), 4);
  ocl::TreeTensor x(x_structure, vs);

  x.set(ocl::Matrix({1,2,3,4,5,6,7,8,9,10}));
  ocl::test::assertEqual(x.get("x1").data(), {{1,2},{9,10}});

  ocl::test::assertEqual(x.get("x1").slice({0},{0}).data(), {{1},{9}});
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

  ocl::test::assertEqual( state.get("p").data(),   {{100, 0, -50}} );
  ocl::test::assertEqual( state.get("R").data(),   {ocl::Matrix::Eye(3).full()} );
  ocl::test::assertEqual( state.get("v").data(),   {{20, 0, 0}} );
  ocl::test::assertEqual( state.get("w").data(),   {{0, 1, 0.1}} );
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

  v.get("x").get("p");

  // v.get("x").get("p").set(ocl::Matrix({100, 0, -50}));
  // v.get("R").set(ocl::Matrix::Eye(3));
  // v.get("v").set(ocl::Matrix({20, 0, 0}));
  // v.get("w").set(ocl::Matrix({0, 1, 0.1}));

  //
  // ocl::test::assertEqual( v.get("x").slice(1, ocl::all(v.get("x"), 1)).at(linspace(3,5)).data(), {100, 100, 100});
  //
  // v.get("x").get("R").set(ocl::Matrix::Eye(3));
  //
  // ocl::test::assertEqual(v.get("x").get("R").data(), repeat(ocl::Matrix::Eye(3), 6).data() );
  // ocl::test::assertEqual(v.get("x").get("R").at(1).data(), ocl::Matrix::Eye(3).data() );
  //
  // v.get("x").get("R").set(ocl::Matrix::One(3,3));
  // ocl::test::assertEqual(v.get("x").get("R").data(), ocl::repeat(ocl::Matrix::One(3,3),6) );
  //
  // ocl::test::assertEqual(v.get("x").get("p").at(1).data(), {100, 0, 50});
  // v.get("x").at(ocl::end(v.get("x")).set(linspace(2,19));
  //
  // ocl::test::assertEqual(v.get("x").at(ocl::end(v.get("x")).slice(2,1).data(),3);
  //
  // ocl::TreeTensor xend = v.get("x").at( ocl::end(v.get("x")) );
  // ocl::test::assertEqual( xend.slice( ocl::end(v.get("x"), 0)).data(), {19} );
  // ocl::test::assertEqual( xend.slice(2, 1)..data(), {3} );
  //
  // v.str();
  // v.get("x").str();
  // v.get("x").get("R").str();
}
