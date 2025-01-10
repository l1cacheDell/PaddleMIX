from gzip import _PaddedFile
import paddlemix
import paddle
import numpy as np

import time

from typing import Any, List, Literal, Optional, Tuple, Union

def precision_diff(quant_o: paddle.Tensor, fa2_o: paddle.Tensor):
    x, xx = paddle.cast(quant_o, dtype='float32'), paddle.cast(fa2_o, dtype='float32')
    # 重塑张量并计算余弦相似度
    x_reshaped = paddle.reshape(x, [1, -1])
    xx_reshaped = paddle.reshape(xx, [1, -1])
    sim = paddle.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (paddle.abs(x - xx).sum() / paddle.abs(xx).sum()).item()
    
    return sim, l1
    
    
if __name__ == "__main__":
    import os
    # os.environ["INFERENCE_OPTIMIZE"] = "True"
    # os.environ["INFERENCE_OPTIMIZE_TRITON"] = "True"
    batch_size = 2
    num_heads = 24
    seq_len = 1375
    head_dim = 64
    q = paddle.to_tensor(paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim)), dtype="float16", place=paddle.CUDAPlace(0))
    k = paddle.to_tensor(paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim)), dtype="float16", place=paddle.CUDAPlace(0))
    v = paddle.to_tensor(paddle.randn(shape=[batch_size, seq_len, num_heads, head_dim]), dtype="float16", place=paddle.CUDAPlace(0))
    # q_npy = np.load("inputs/q.npy",)
    # q = paddle.to_tensor(q_npy, dtype=paddle.float16)
    # k_npy = np.load("inputs/k.npy",)
    # k = paddle.to_tensor(k_npy, dtype=paddle.float16)
    # v_npy = np.load("inputs/v.npy",)
    # v = paddle.to_tensor(v_npy, dtype=paddle.float16)

    sm_scale = 1 / (head_dim ** 0.5)
    
    # start = time.monotonic()
    for i in range(10): o = paddlemix.triton_ops.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="NHD", is_casual=False, sm_scale=sm_scale, smooth_k=True, return_lse=False)
    paddle.device.cuda.synchronize()
    # end = time.monotonic()
    
    # print("our kernel time: ", end - start)
    
    # start = time.monotonic()
    for i in range(100): o2 = paddle.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    
    paddle.device.cuda.synchronize()
    # end = time.monotonic()
    # print("FA2 kernel time: ", end - start)
    
    # print("our kernel: ", o[0, 0, 0, :])
    # print("FA2 kernel: ", o2[0, 0, 0, :])
    
    sim, l1 = precision_diff(o, o2)
    print(f"sim: {sim}, l1: {l1}")
    print(paddle.max(o2-o, axis=[0,1,2,3]))
    print((o2-o)[0, :, 0, 0])