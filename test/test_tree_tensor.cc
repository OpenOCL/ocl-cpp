#include <utils/testing.h>
#include "tensor/tree.h"


TEST(testTreeTensor, aThreeVariables)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});
  tb.add("x1", {1,2});

  ocl::Tree x_structure = tb.tree();

  ocl::ValueStorage data(x_structure.size(), 4);
  ocl::TreeTensor x(x_structure, data);

  x.slice(ocl::all(x, 0), 1).set(ocl::linspace(1,10));

  ocl::testAssertEqual(x.data, ocl::linspace(1,10));
  ocl::testAssertEqual(x.get("x1").data, {1,9,2,10});

  ocl::testAssertEqual(x.get("x1").slice(1,1).data, {1,9});

}

TEST(testTreeTensor, bStateTensor)
{
  ocl::TreeBuilder tb_x;
  tb_x.add("p",{3,1});
  tb_x.add("R",{3,3});
  tb_x.add("v",{3,1});
  tb_x.add("w",{3,1});

  state = OclTensor.create(tb_x,0);

  state.get("R").set(ocl::eye(3));
  state.get("p").set({100, 0, -50});
  state.get("v").set({20, 0, 0});
  state.get("w").set({0, 1, 0.1});

  ocl::testAssertEqual( state.get("R").value,   ocl::eye(3) );
  ocl::testAssertEqual( state.get("p").value,   {100, 0, -50} );
  ocl::testAssertEqual( state.get("v").value,   {20, 0, 0} );
  ocl::testAssertEqual( state.get("w").value,   {0, 1, 0.1} );

  ocl::testAssertEqual( state.size(),   {18, 1} );
}

TEST(testTreeTensor, bOcpTensor)
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
  tb_ocp.addRepeated({"x","u"},{x, u}, 5);
  tb_ocp.add("x",x);

  ocp_tree = tb_ocp.tree();

  vs = ValueStorage(ocp_tree.size(), 0);

  v = OclTensor.create(ocp_tree, vs);
  v.get("R").set(ocl::eye(3));
  v.get("p").set({100, 0, -50});
  v.get("v").set({20, 0, 0});
  v.get("w").set({0, 1, 0.1});

  assertSqueezeEqual( v.x.p(1,:,4:6).value, {100;100;100});

  v.get("x").get("R").set(eye(3));

  assertEqual(v.x.get("R").value,   shiftdim(num2cell(repmat(eye(3),1,1,6), 1:2), 1)    );
  assertEqual(v.x.R(:,:,1).value,eye(3));

  v.get("x").get("R").set(ones(3,3));
  assertEqual(v.x.R.value,   shiftdim(num2cell(repmat(ones(3),1,1,6), 1:2), 1));

  assertEqual(v.x.p(:,:,1).value,{100;0;50});
  assertEqual(v(":").value,v.value);
  v.x(:,end).set((2:19)");

  assertEqual(v.x(:,end).slice(2).value,3);

  xend = v.x(:,end);
  assertEqual(xend(end).value,19);
  assertEqual(xend(2).value,3);
  assertEqual(xend(end).value,19);

  v.str();
  v.x.str();
  v.x.R.str();

}
