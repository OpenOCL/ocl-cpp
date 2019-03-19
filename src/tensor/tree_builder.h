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
#ifndef OCLCPP_OCL_TREEBUILDER_H_
#define OCLCPP_OCL_TREEBUILDER_H_

#include "utils/assertions.h"   // assertEqual
#include "utils/functions.h"    // prod, merge
#include "tensor/tree.h"        // Tree

namespace ocl {

class TreeBuilder
{
public:

  TreeBuilder() : _len(0), _tree(Tree()) { }

  void add(const std::string& id, const int length = 1) {
    add(id, {length, 1});
  }

  void add(const std::string& id, const std::vector<int>& shape = {1,1})
  {
    int N = prod(shape);
    Tree tree = Tree( Tree::Branches(), shape, {range(_len, N+_len)} );
    _len += N;
    this->addTree(id, tree);
  }

  void add(const std::string& id, const Tree& tree)
  {
    int N = tree.size();
    Tree t = Tree( tree._branches, tree.shape(), {range(_len, N+_len)} );
    _len += N;
    addTree(id, t);

  }

  void addRepeated(const std::vector<std::string>& ids, const std::vector<Tree>& trees, const int N)
  {
    assertEqual(ids.size(), trees.size(), "Number of ids must correspond to the number of trees to add.");

    for (int i=0; i<N; i++) {
      for (unsigned int j=0; j<ids.size(); j++) {
        add(ids[j], trees[j]);
      }
    }
  }

  void addTree(const std::string& id, const Tree& tree)
  {
    auto it =  _tree._branches.find(id);
    if (it == _tree._branches.end())
    {
      std::pair<std::string, Tree> el(id, tree);
      _tree._branches.insert(el);
    }
    else
    {
      Tree& branch = _tree._branches.at(id);

      // append all indizes of
      branch._indizes = merge<std::vector<int>>(branch._indizes, tree._indizes);
    }
    _tree._shape = {_len, 1};
    _tree._indizes = {range(0, _len)};
  }

  // Returns a reference to the tree object which the tree builder owns
  Tree tree() { return _tree; };

 private:
  int _len;
  Tree _tree;
};

} // namespace ocl
#endif // OCLCPP_OCL_TREEBUILDER_H_
