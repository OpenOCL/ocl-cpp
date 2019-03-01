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

namespace ocl {

class TreeBuilder : public RootNode
{
 public:
  TreeBuilder();

  void add(const String& id, const int length);
  void add(const String& id, const Size& size);
  void add(const String& id, const RootNode& node);

  void addRepeated(const std::vector<String>& ids, const std::vector<RootNode>& nodes, const int N);

 private:
  int len;
};

} // namespace ocl
#endif OCLCPP_OCL_TREEBUILDER_H_
