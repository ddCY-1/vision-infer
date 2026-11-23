#include "vt/conv.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vt {

Tensor maxpool2d(const Tensor& x, int k, int stride) {
    if (k < 1) throw std::invalid_argument("maxpool2d: k 必须 >= 1");
    if (stride == 0) stride = k;
    if (x.n != 1) throw std::invalid_argument("maxpool2d: 暂时只支持 batch=1");

    const int oh = (x.h - k) / stride + 1;
    const int ow = (x.w - k) / stride + 1;
    if (oh <= 0 || ow <= 0) throw std::invalid_argument("maxpool2d: 输出尺寸非正");

    Tensor y(1, x.c, oh, ow);
    for (int ci = 0; ci < x.c; ++ci) {
        for (int yi = 0; yi < oh; ++yi) {
            for (int xi = 0; xi < ow; ++xi) {
                float m = std::numeric_limits<float>::lowest();
                for (int ky = 0; ky < k; ++ky)
                    for (int kx = 0; kx < k; ++kx)
                        m = std::max(m, x.at(0, ci, yi * stride + ky, xi * stride + kx));
                y.at(0, ci, yi, xi) = m;
            }
        }
    }
    return y;
}

} // namespace vt
