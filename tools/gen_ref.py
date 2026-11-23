#!/usr/bin/env python3
"""用 numpy 生成对拍数据：同一组输入/权重，算出 conv/leaky/pool 的参考输出。

C++ 那边只负责读 bin + 复现，数值对不上就是 bug。
"""
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20261123)

N, C, H, W = 1, 3, 9, 9
O, KH, KW = 4, 3, 3
S, P = 2, 1

x = rng.standard_normal((N, C, H, W), dtype=np.float32)
w = rng.standard_normal((O, C, KH, KW), dtype=np.float32)
b = rng.standard_normal((O,), dtype=np.float32)

OH = (H + 2 * P - KH) // S + 1
OW = (W + 2 * P - KW) // S + 1
xp = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)))

cols = np.zeros((N, OH * OW, C * KH * KW), dtype=np.float32)
for yi in range(OH):
    for xi in range(OW):
        for ci in range(C):
            for ky in range(KH):
                for kx in range(KW):
                    cols[0, yi * OW + xi, (ci * KH + ky) * KW + kx] = xp[0, ci, yi * S + ky, xi * S + kx]

conv = (cols @ w.reshape(O, -1).T + b).reshape(N, OH, OW, O).transpose(0, 3, 1, 2)
conv = np.ascontiguousarray(conv, dtype=np.float32)

lk = np.where(conv < 0, conv * np.float32(0.1), conv).astype(np.float32)

K, PS = 2, 2
PH, PW = (OH - K) // PS + 1, (OW - K) // PS + 1
pool = np.zeros((N, O, PH, PW), dtype=np.float32)
for ni in range(N):
    for ci in range(O):
        for yi in range(PH):
            for xi in range(PW):
                pool[ni, ci, yi, xi] = lk[ni, ci, yi * PS:yi * PS + K, xi * PS:xi * PS + K].max()

for name, arr in [("x", x), ("w", w), ("b", b), ("conv", conv), ("leaky", lk), ("pool", pool)]:
    (OUT / f"{name}.bin").write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())

meta = dict(N=N, C=C, H=H, W=W, O=O, KH=KH, KW=KW, S=S, P=P,
            OH=OH, OW=OW, K=K, PS=PS, PH=PH, PW=PW)
(OUT / "shape.txt").write_text("\n".join(f"{k} {v}" for k, v in meta.items()) + "\n")
print("ref written to", OUT)
print({k: tuple(v.shape) for k, v in dict(x=x, w=w, b=b, conv=conv, leaky=lk, pool=pool).items()})
