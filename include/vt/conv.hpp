#pragma once
#include "vt/tensor.hpp"

namespace vt {

// 普通卷积（不是深度可分离），NCHW，权重 OIHW。pad 为四周补零宽度。
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b,
              int stride = 1, int pad = 0);

// 最大池化，窗口 k×k，stride 默认等于 k。
Tensor maxpool2d(const Tensor& x, int k, int stride = 0);

} // namespace vt
