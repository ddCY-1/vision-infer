// conv2d 基线 benchmark：跑几组典型尺寸，输出 ms / GFLOPS / im2col 内存占用。
// 目的不是跑分，是给后面每一次优化留一个可比对的锚点。
#include "vt/conv.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    int c, h, w;   // 输入
    int o, k;      // 输出通道、卷积核
    int s, p;
};

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

void fill_random(vt::Tensor& t, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    for (float& v : t.d) v = nd(rng);
}

} // namespace

int main() {
    const std::vector<Case> cases{
        {"stem     3x224->64", 3, 224, 224, 64, 3, 1, 1},
        {"mid     32x112->64", 32, 112, 112, 64, 3, 1, 1},
        {"deep    64x56 ->128", 64, 56, 56, 128, 3, 1, 1},
        {"pointwise 128x28->128", 128, 28, 28, 128, 1, 1, 0},
    };

    std::ofstream json("bench_results.json");
    json << "{\n  \"build\": \"Release (-O3)\",\n  \"cases\": [\n";

    std::printf("%-24s %10s %10s %10s %8s\n", "case", "ms/iter", "GFLOPS", "im2col MB", "runs");
    std::printf("%s\n", std::string(68, '-').c_str());

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& cs = cases[i];
        vt::Tensor x(1, cs.c, cs.h, cs.w);
        vt::Tensor wgt(cs.o, cs.c, cs.k, cs.k);
        vt::Tensor bias(1, cs.o, 1, 1);
        fill_random(x, 1 + unsigned(i));
        fill_random(wgt, 7 + unsigned(i));
        fill_random(bias, 13 + unsigned(i));

        // 输出尺寸决定 im2col 的膨胀量
        const int oh = (cs.h + 2 * cs.p - cs.k) / cs.s + 1;
        const int ow = (cs.w + 2 * cs.p - cs.k) / cs.s + 1;
        const double cols_mb = double(oh) * ow * cs.c * cs.k * cs.k * 4.0 / (1024 * 1024);

        volatile float sink = 0.0f;   // 别让编译器把结果优化掉
        sink = vt::conv2d(x, wgt, bias, cs.s, cs.p).d[0];   // warmup
        const int runs = (cs.h * cs.w * cs.c > 200000) ? 3 : 5;

        double best = 1e18;
        for (int r = 0; r < runs; ++r) {
            const auto t0 = Clock::now();
            const auto y = vt::conv2d(x, wgt, bias, cs.s, cs.p);
            const auto t1 = Clock::now();
            sink = y.d[y.d.size() / 2];
            best = std::min(best, Ms(t1 - t0).count());
        }
        (void)sink;

        const double flops = 2.0 * cs.o * oh * ow * cs.c * cs.k * cs.k;
        const double gflops = flops / (best * 1e6);
        std::printf("%-24s %10.2f %10.3f %10.2f %8d\n", cs.name, best, gflops, cols_mb, runs);

        json << "    {\"name\": \"" << cs.name << "\", \"ms\": " << best
             << ", \"gflops\": " << gflops << ", \"im2col_mb\": " << cols_mb
             << ", \"shapes\": {\"c\": " << cs.c << ", \"h\": " << cs.h << ", \"w\": " << cs.w
             << ", \"o\": " << cs.o << ", \"k\": " << cs.k << "}}"
             << (i + 1 < cases.size() ? ",\n" : "\n");
    }
    json << "  ]\n}\n";
    std::puts("\n-> bench_results.json");
    return 0;
}
