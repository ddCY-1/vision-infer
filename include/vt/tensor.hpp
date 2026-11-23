#pragma once
#include <cassert>
#include <cstddef>
#include <vector>

namespace vt {

// 最小 NCHW 张量：只有数据和形状，够用来跑前向。
struct Tensor {
    int n = 0, c = 0, h = 0, w = 0;
    std::vector<float> d;

    Tensor() = default;
    Tensor(int n_, int c_, int h_, int w_)
        : n(n_), c(c_), h(h_), w(w_), d(std::size_t(n_) * c_ * h_ * w_, 0.0f) {}

    std::size_t size() const { return d.size(); }
    int index(int ni, int ci, int hi, int wi) const {
        return ((ni * c + ci) * h + hi) * w + wi;
    }
    float& at(int ni, int ci, int hi, int wi) { return d[index(ni, ci, hi, wi)]; }
    float at(int ni, int ci, int hi, int wi) const { return d[index(ni, ci, hi, wi)]; }
};

inline Tensor relu(const Tensor& x) {
    Tensor y = x;
    for (float& v : y.d) if (v < 0.0f) v = 0.0f;
    return y;
}

inline Tensor leaky_relu(const Tensor& x, float slope = 0.1f) {
    Tensor y = x;
    for (float& v : y.d) if (v < 0.0f) v *= slope;
    return y;
}

} // namespace vt
