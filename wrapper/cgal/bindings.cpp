//
// Created by xingyu on 5/29/23.
//
#include "core/utils.h"
#include "intersection.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("self_intersections", &self_intersections);
}