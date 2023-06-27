//
// Created by xingyu on 5/29/23.
//

#pragma once

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include "core/utils.h"
#include "core/cuda_utils.h"

typedef CGAL::Simple_cartesian<float> K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef Mesh::Vertex_index vertex_descriptor;
typedef Mesh::Face_index face_descriptor;

void tensor_to_cgal_mesh(const at::Tensor& xyz, const at::Tensor& faces, Mesh& mesh);