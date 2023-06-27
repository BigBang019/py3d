//
// Created by xingyu on 5/26/23.
//

#include "arap.h"

void arap_deform_cpu(at::Tensor xyz, at::Tensor faces, at::Tensor handles, at::Tensor moves,
                     at::Tensor new_xyz);


at::Tensor arap_deform(at::Tensor xyz, at::Tensor faces, at::Tensor handles, at::Tensor moves) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int M = faces.size(1);

    at::Tensor new_xyz = xyz.clone();

    if (xyz.is_cuda()) { // gpu
        TORCH_CHECK(false, "gpu not supported");
    } else {
        arap_deform_cpu(xyz, faces, handles, moves, new_xyz);
    }
    return new_xyz;
}