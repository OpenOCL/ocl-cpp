#include <utils/testing.h>
#include "tensor/tree_builder.h"


TEST(testTree, aTwoVariables)
{
  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});

  ocl::Tree x = tb.tree();

  ocl::test::assertEqual(x.get("x1").indizes(),{{0,1}});
  ocl::test::assertEqual(x.get("x2").indizes(),{{2,3,4,5,6,7}});
  // ocl::test::assertEqual(x.shape(),{8,1});
}

// TEST(testTree, bShape)
// {
//   ocl::TreeBuilder tb;
//   tb.add("x1",{1,8});
//
//   ocl::Tree x = tb.tree();
//
//   ocl::test::assertEqual(x.shape(), {8,1});
// }
//
// TEST(testTree, cRepeatedVar) {
//   ocl::TreeBuilder tb;
//   tb.add("x1",{1,3});
//   tb.add("x2",{3,2});
//   tb.add("x1",{1,3});
//
//   ocl::Tree x = tb.tree();
//
//   ocl::test::assertEqual(x.get("x1").indizes(),{{1,2,3},{10,11,12}});
//   ocl::test::assertEqual(x.get("x2").indizes(),{{4,5,6,7,8,9}});
// }
//
// TEST(testTree, dSubTree)
// {
//   ocl::TreeBuilder tb_u;
//   tb_u.add("x1",{1,3});
//   tb_u.add("x3",{3,3});
//   tb_u.add("x1",{1,3});
//   tb_u.add("x3",{3,3});
//
//   ocl::Tree u = tb_u.tree();
//
//   ocl::TreeBuilder tb_x;
//   tb_x.add("x1",{1,3});
//   tb_x.add("u",u);
//   tb_x.add("u",u);
//   tb_x.add("x2",{3,2});
//   tb_x.add("x1",{1,3});
//
//   ocl::Tree x = tb_x.tree();
//
//   ocl::test::assertEqual(x.get("u").get("x1").indizes(), {{4,5,6},{16,17,18},{28,29,30},{40,41,42}} );
//
//   ocl::Tree r = x.get("u").at(1).get("x1");
//   ocl::test::assertEqual(r.indizes(), {{4,5,6},{16,17,18}} );
// }
