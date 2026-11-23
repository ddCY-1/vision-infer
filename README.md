# vision-infer

从零用 C++ 写视觉推理：先手写算子，再接真实模型。

## 现状

| 模块 | 状态 | 验证方式 |
|---|---|---|
| `Tensor` (NCHW) | ✅ | — |
| `conv2d`（im2col + GEMM 朴素版） | ✅ | 与 numpy 对拍 max\|diff\| = 1.9e-06 |
| `leaky_relu` | ✅ | 1.2e-06 |
| `maxpool2d` | ✅ | 9.5e-07 |

对拍数据由 `tools/gen_ref.py`（numpy，固定 seed 20261123）生成到 `tests/data/`。
换 seed 或换尺寸重跑一遍，就能验证 C++ 实现是否还正确。

## 跑起来

```bash
python3 tools/gen_ref.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/test_layers          # 需要从仓库根目录跑
```

## 已知限制（有意为之，一步步补）

- 只支持 batch=1
- 只支持单线程、朴素 im2col（没有分块、没有 SIMD）
- 没有 padding 以外的特殊卷积（dilation/group/depthwise 都没做）

## 下一步

1. batch > 1
2. 手写算子的 benchmark（ns/op、GFLOPS），有基线才知道优化有没有用
3. 读 darknet 的 `src/convolutional_layer.c` —— 真实工业级 C 实现怎么写的
