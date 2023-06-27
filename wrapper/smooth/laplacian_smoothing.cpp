//
// Created by xingyu on 5/24/23.
//

#include "laplacian_smoothing.h"

#include "core/cuda_utils.h"
#include "core/utils.h"

void implicit_laplacian_smoothing_cpu(at::Tensor vertices, at::Tensor faces, float alpha, at::Tensor new_vertices);

//void implicit_laplacian_smoothing_gpu(at::Tensor xyz, at::Tensor faces, float alpha, at::Tensor new_vertices) {
//    TORCH_CHECK(false, "gpu not supported");
//}

at::Tensor implicit_laplacian_smoothing(at::Tensor xyz, at::Tensor faces, float alpha) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int M = faces.size(1);

    at::Tensor new_vertices = torch::zeros({B, N, 3}, at::device(xyz.device()).dtype(at::ScalarType::Float));
    if (xyz.is_cuda()) { //gpu
        TORCH_CHECK(false, "gpu not supported");
    } else {
        implicit_laplacian_smoothing_cpu(xyz, faces, alpha, new_vertices);
    }
    return new_vertices;
}