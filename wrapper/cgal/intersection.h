//
// Created by xingyu on 5/29/23.
//
#pragma once
#include "core/cuda_utils.h"
#include "core/utils.h"

at::Tensor self_intersections(at::Tensor xyz, at::Tensor faces);