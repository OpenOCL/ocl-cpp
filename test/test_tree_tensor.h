#include <utils/testing.h>
#include "tensor/tree_builder.h"
#include "tensor/tree_tensor.h"

TEST(testTreeTensor, aThreeVariables)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree x_structure = tb.tree();

  ocl::ValueStorage vs(x_structure.size(), 4);
  ocl::TreeTensor x(x_structure, vs);

  // x.slice(ocl::all(x, 0), 1).set(ocl::linspace(1,10));
  //
  // ocl::test::assertEqual(x.data(), ocl::linspace(1,10));
  // ocl::test::assertEqual(x.get("x1").data(), {1,9,2,10});
  //
  // ocl::test::assertEqual(x.get("x1").slice(1,1).data(), {1,9});

}

// TEST(testTreeTensor, bStateTensor)
// {
//   ocl::TreeBuilder tb_x;
//   tb_x.add("p",{3,1});
//   tb_x.add("R",{3,3});
//   tb_x.add("v",{3,1});
//   tb_x.add("w",{3,1});
//
//   ocl::Tree x_structure = tb.tree();
//   ocl::ValueStorage vs(x_structure.size(), 0);
//   state = OclTensor::create(tb_x, vs);
//
//   state.get("R").set(ocl::Matrix::Eye(3));
//   state.get("p").set({100, 0, -50});
//   state.get("v").set({20, 0, 0});
//   state.get("w").set({0, 1, 0.1});
//
//   ocl::test::assertEqual( state.get("R").data(),   ocl::Matrix::Eye(3).data() );
//   ocl::test::assertEqual( state.get("p").data(),   {100, 0, -50} );
//   ocl::test::assertEqual( state.get("v").data(),   {20, 0, 0} );
//   ocl::test::assertEqual( state.get("w").data(),   {0, 1, 0.1} );
//
//   ocl::test::assertEqual( state.size(),   {18, 1} );
// }
//
// TEST(testTreeTensor, bOcpTensor)
// {
//   ocl::TreeBuilder tb_x;
//   tb_x.add("p",{3,1});
//   tb_x.add("R",{3,3});
//   tb_x.add("v",{3,1});
//   tb_x.add("w",{3,1});
//
//   ocl::TreeBuilder tb_u;
//   tb_u.add("elev",{1,1});
//   tb_u.add("ail",{1,1});
//
//   ocl::TreeBuilder tb_ocp;
//   tb_ocp.addRepeated({"x","u"},{x, u}, 5);
//   tb_ocp.add("x",x);
//
//   ocp_tree = tb_ocp.tree();
//
//   vs = ValueStorage(ocp_tree.size(), 0);
//
//   v = OclTensor.create(ocp_tree, vs);
//   v.get("R").set(ocl::Matrix::Eye(3));
//   v.get("p").set({100, 0, -50});
//   v.get("v").set({20, 0, 0});
//   v.get("w").set({0, 1, 0.1});
//
//   ocl::test::assertEqual( v.get("x").slice(1, ocl::all(v.get("x"), 1)).at(linspace(3,5)).data(), {100, 100, 100});
//
//   v.get("x").get("R").set(ocl::Matrix::Eye(3));
//
//   ocl::test::assertEqual(v.get("x").get("R").data(), repeat(ocl::Matrix::Eye(3), 6).data() );
//   ocl::test::assertEqual(v.get("x").get("R").at(1).data(), ocl::Matrix::Eye(3).data() );
//
//   v.get("x").get("R").set(ocl::Matrix::One(3,3));
//   ocl::test::assertEqual(v.get("x").get("R").data(), ocl::repeat(ocl::Matrix::One(3,3),6) );
//
//   ocl::test::assertEqual(v.get("x").get("p").at(1).data(), {100, 0, 50});
//   v.get("x").at(ocl::end(v.get("x")).set(linspace(2,19));
//
//   ocl::test::assertEqual(v.get("x").at(ocl::end(v.get("x")).slice(2,1).data(),3);
//
//   ocl::TreeTensor xend = v.get("x").at( ocl::end(v.get("x")) );
//   ocl::test::assertEqual( xend.slice( ocl::end(v.get("x"), 0)).data(), {19} );
//   ocl::test::assertEqual( xend.slice(2, 1)..data(), {3} );
//
//   v.str();
//   v.get("x").str();
//   v.get("x").get("R").str();
// }
