//
// Created by xingyu on 5/24/23.
//

#pragma once
#include <torch/extension.h>

at::Tensor implicit_laplacian_smoothing(at::Tensor xyz, at::Tensor faces, float alpha);
