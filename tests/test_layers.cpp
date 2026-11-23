#include "vt/conv.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::string DATA = "tests/data/";

std::vector<float> load(const std::string& name) {
    std::ifstream f(DATA + name + ".bin", std::ios::binary);
    if (!f) throw std::runtime_error("打不开 " + name + ".bin（先跑 tools/gen_ref.py）");
    std::vector<float> v;
    float buf;
    while (f.read(reinterpret_cast<char*>(&buf), sizeof(float))) v.push_back(buf);
    return v;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::runtime_error("长度不一致: " + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

int failures = 0;

void check(const char* label, float diff) {
    const bool ok = diff < 1e-4f;
    std::printf("%-12s max|diff| = %.3e  %s\n", label, double(diff), ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

} // namespace

int main() {
    const int N = 1, C = 3, H = 9, W = 9, O = 4, KH = 3, KW = 3, S = 2, P = 1;
    const int OH = 5, OW = 5, K = 2, PS = 2, PH = 2, PW = 2;

    vt::Tensor x(N, C, H, W), w(O, C, KH, KW), b(1, O, 1, 1);
    x.d = load("x");
    w.d = load("w");
    b.d = load("b");

    const auto y_conv = vt::conv2d(x, w, b, S, P);
    if (y_conv.h != OH || y_conv.w != OW)
        throw std::runtime_error("conv 输出尺寸错了");
    check("conv2d", max_abs_diff(y_conv.d, load("conv")));

    const auto y_lk = vt::leaky_relu(y_conv, 0.1f);
    check("leaky_relu", max_abs_diff(y_lk.d, load("leaky")));

    const auto y_pool = vt::maxpool2d(y_lk, K, PS);
    if (y_pool.h != PH || y_pool.w != PW)
        throw std::runtime_error("maxpool 输出尺寸错了");
    check("maxpool2d", max_abs_diff(y_pool.d, load("pool")));

    if (failures) {
        std::printf("%d 项失败\n", failures);
        return 1;
    }
    std::puts("全部通过：C++ 实现与 numpy 参考一致");
    return 0;
}
