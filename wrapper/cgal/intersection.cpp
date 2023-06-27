//
// Created by xingyu on 5/29/23.
//
#include "intersection.h"

#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include "core/cuda_utils.h"
#include "core/utils.h"
#include "utils.h"

namespace PMP = CGAL::Polygon_mesh_processing;

at::Tensor self_intersections(at::Tensor vertices, at::Tensor triangles) {
    CHECK_IS_FLOAT(vertices);
    CHECK_IS_INT(triangles);
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(triangles);

    if (vertices.is_cuda()) { // gpu
        TORCH_CHECK(false, "gpu not supported");
    } else {
        Mesh mesh;
        tensor_to_cgal_mesh(vertices, triangles, mesh);
        std::vector<std::pair<face_descriptor, face_descriptor> > intersected_tris;
        PMP::self_intersections<CGAL::Parallel_if_available_tag>(faces(mesh), mesh, std::back_inserter(intersected_tris));
        std::vector<int> face_pair_vec;
        at::Tensor face_pairs = torch::zeros({intersected_tris.size(), 2}, at::device(at::kCPU).dtype(at::ScalarType::Int));
        for (int i = 0; i < intersected_tris.size(); ++i) {
            std::pair<face_descriptor, face_descriptor> face_pair = intersected_tris[i];
            *face_pairs[i][0].data_ptr<int>() = face_pair.first.idx();
            *face_pairs[i][1].data_ptr<int>() = face_pair.second.idx();
        }
        return face_pairs;
    }
}