//
// Created by xingyu on 5/29/23.
//

#include "utils.h"


void tensor_to_cgal_mesh(const at::Tensor& xyz, const at::Tensor& faces, Mesh& mesh) {
    const int N = xyz.size(0);
    const int M = faces.size(0);
    std::vector<vertex_descriptor> vertices(N);
    for (int i = 0; i < N; ++i) {
        vertices[i] = mesh.add_vertex(
                K::Point_3(xyz[i][0].item<float>(), xyz[i][1].item<float>(), xyz[i][2].item<float>()));
    }
    for (int i = 0; i < M; ++i) {
        face_descriptor f = mesh.add_face(\
            vertices[faces[i][0].item<int>()], \
            vertices[faces[i][1].item<int>()], \
            vertices[faces[i][2].item<int>()] \
        );
        if (f == Mesh::null_face()) {
            ERROR("The face could not be added because of an orientation error.");
            f = mesh.add_face(\
                vertices[faces[i][0].item<int>()], \
                vertices[faces[i][2].item<int>()], \
                vertices[faces[i][1].item<int>()] \
            );
            assert(f != Mesh::null_face());
        }
    }
}