#include "vt/conv.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vt {

namespace {

// im2col：把每个输出位置的 C*KH*KW 邻域摊平成一行，卷积退化成 GEMM。
// 先写清楚正确版本，性能优化（分块/多线程）留到后面再做。
std::vector<float> im2col(const Tensor& x, int kh, int kw, int stride, int pad,
                          int oh, int ow) {
    const int K = x.c * kh * kw;
    std::vector<float> cols(std::size_t(oh) * ow * K, 0.0f);
    for (int yi = 0; yi < oh; ++yi) {
        for (int xi = 0; xi < ow; ++xi) {
            float* dst = &cols[(std::size_t(yi) * ow + xi) * K];
            for (int ci = 0; ci < x.c; ++ci) {
                for (int ky = 0; ky < kh; ++ky) {
                    const int sy = yi * stride + ky - pad;
                    for (int kx = 0; kx < kw; ++kx) {
                        const int sx = xi * stride + kx - pad;
                        float v = 0.0f;
                        if (sy >= 0 && sy < x.h && sx >= 0 && sx < x.w)
                            v = x.at(0, ci, sy, sx);   // phase 1 先只支持 batch=1
                        *dst++ = v;
                    }
                }
            }
        }
    }
    return cols;
}

} // namespace

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b,
              int stride, int pad) {
    if (x.n != 1) throw std::invalid_argument("conv2d: 暂时只支持 batch=1");
    if (w.c != x.c) throw std::invalid_argument("conv2d: 输入通道数与权重不符");
    if (b.size() != std::size_t(w.n)) throw std::invalid_argument("conv2d: bias 长度不符");
    if (stride < 1) throw std::invalid_argument("conv2d: stride 必须 >= 1");

    const int oh = (x.h + 2 * pad - w.h) / stride + 1;
    const int ow = (x.w + 2 * pad - w.w) / stride + 1;
    if (oh <= 0 || ow <= 0) throw std::invalid_argument("conv2d: 输出尺寸非正");

    const int K = x.c * w.h * w.w;
    const auto cols = im2col(x, w.h, w.w, stride, pad, oh, ow);

    Tensor y(1, w.n, oh, ow);
    for (int oc = 0; oc < w.n; ++oc) {
        const float* wrow = &w.d[std::size_t(oc) * K];
        for (int p = 0; p < oh * ow; ++p) {
            const float* crow = &cols[std::size_t(p) * K];
            float acc = b.d[std::size_t(oc)];
            for (int k = 0; k < K; ++k) acc += wrow[k] * crow[k];
            y.d[std::size_t(oc) * oh * ow + p] = acc;
        }
    }
    return y;
}

} // namespace vt
