//
// Created by xingyu on 5/25/23.
//
#include "laplacian_smoothing.h"
#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_laplacian_smoothing", &implicit_laplacian_smoothing);
}