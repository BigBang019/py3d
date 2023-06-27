//
// Created by xingyu on 5/28/23.
//
#include "arap.h"
#include "core/utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("arap_deform", &arap_deform);
}