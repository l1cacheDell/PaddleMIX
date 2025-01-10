import paddle
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from typing import Any, List, Literal, Optional, Tuple, Union

from .triton_utils import get_dtype_str, paddle_use_triton, rendering_common_template

########################### sage attention ###############################
@paddle_use_triton(
    key=["1"]
)
def sageattn_quant_per_block_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz, 
    stride_ih, 
    stride_in,
    stride_oz, 
    stride_oh, 
    stride_on,
    stride_sz, 
    stride_sh,
    sm_scale,
    Grid,                   # grid num, through compiling
    h_attn,                 # grid num, through compiling
    bsz,                    # grid num, through compiling
    C: tl.constexpr,
    BLK: tl.constexpr
):
    off_blk = tl.program_id(axis=0)
    off_h = tl.program_id(axis=1)
    off_b = tl.program_id(axis=2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x_data = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x_data = x_data.to(tl.float32)
    x_data *= sm_scale
    scale = tl.max(tl.abs(x_data)) / 127.
    x_int8 = x_data / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)
    
# note: here we need to do one single operation, instead of fused two.
# reference: quant_per_block.py
def sageattn_quant_per_block_int8(x, 
                                km=None, 
                                BLKQ=128, BLKK=64,
                                sm_scale=1.0, 
                                tensor_layout="HND", q_or_k="q"):
    

    if km is not None and q_or_k == "k":
        x = x - km
        
    if tensor_layout == "HND":
        b, h_attn, seq_len, head_dim = x.shape

        # there is no stride in static mode, so we need to compute it manually
        stride_iz, stride_ih, stride_in = head_dim * seq_len * h_attn, head_dim * seq_len, head_dim * 1
        stride_oz, stride_oh, stride_on = head_dim * seq_len * h_attn, head_dim * seq_len, head_dim * 1
    elif tensor_layout == "NHD":
        b, seq_len, h_attn, head_dim = x.shape
        
        stride_iz, stride_ih, stride_in = head_dim * seq_len * h_attn, head_dim * 1, head_dim * h_attn
        stride_oz, stride_oh, stride_on = head_dim * seq_len * h_attn, head_dim * 1, head_dim * h_attn,
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    L = seq_len
    C = head_dim
    BLK = BLKQ if q_or_k == "q" else BLKK
    gd = BLK
    sm_scale = sm_scale * 1.44269504 if q_or_k == "q" else 1.0

    stride_sz = h_attn * ((seq_len + BLK - 1) // BLK)
    stride_sh =  (seq_len + BLK - 1) // BLK
    
    prepare_attr_for_triton_kernel = """
    auto output_tensor = paddle::empty(x.shape(), paddle::DataType::INT8, x.place());
    
    auto input_tensor = x;
    auto input_shape = x.shape();
        
    // define params
    int b, h_attn, seq_len, head_dim;
    int stride_iz, stride_ih, stride_in;
    int stride_oz, stride_oh, stride_on;
    
    // allocate
    if (tensor_layout == std::string("HND")) {
        // tensor layout unpack
        b = input_shape[0];
        h_attn = input_shape[1];
        seq_len = input_shape[2];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[1];
        stride_in = tensor_strides[2];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[1];
        stride_on = tensor_o_strides[2];
        
    } else if (tensor_layout == std::string("NHD")) {
        // tensor layout unpack
        b = input_shape[0];
        h_attn = input_shape[2];    // reverse
        seq_len = input_shape[1];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[2];    // reverse
        stride_in = tensor_strides[1];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[2];  // reverse
        stride_on = tensor_o_strides[1];
    }
    else {
        throw std::runtime_error("Unsupported tensor layout");
    }
    int BLK = (q_or_k == std::string("q")) ? BLKQ : BLKK;
    auto scale_tensor = paddle::empty({b, h_attn, (seq_len + BLK - 1) / BLK}, 
                                        paddle::DataType::FLOAT32, 
                                        x.place());
    int L = seq_len;
    int stride_sz = scale_tensor.strides()[0];
    int stride_sh = scale_tensor.strides()[1];
    int Grid = BLK;
    int bsz = b;
"""

    op_name = "triton_sageattn_quant_per_block"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_BLK{BLK}_seq{seq_len}_h{h_attn}_dim{head_dim}"
    
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        Output = paddle.empty(x.shape, dtype=paddle.int8)
        Scale = paddle.empty((b, h_attn, (seq_len + BLK - 1) // BLK), dtype='float32')
        # due to compute reasons, output_tensor & scale_tensor has beed defined in above areas, see `prepare_attr_for_triton_kernel`
        prepare_ptr_for_triton_kernel = """
        // prepare tensor
        auto Input = get_tensor_ptr(x);
        auto Output = get_tensor_ptr(output_tensor);
        auto Scale = get_tensor_ptr(scale_tensor);
"""
        return_tensor_names = "output_tensor, scale_tensor"
        
        template_used = rendering_common_template(
            sageattn_quant_per_block_int8, 
            prepare_attr_for_triton_kernel=prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel=prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names
        )
        grid = ("(L + Grid - 1) / Grid", "h_attn", "bsz")
        sageattn_quant_per_block_int8_kernel[(op_name, template_used, grid)](
            Input=x, 
            Output=Output, 
            Scale=Scale, 
            L=L, 
            stride_iz=stride_iz, 
            stride_ih=stride_ih, 
            stride_in=stride_in,
            stride_oz=stride_oz, 
            stride_oh=stride_oh, 
            stride_on=stride_on,
            stride_sz=stride_sz, 
            stride_sh=stride_sh,
            sm_scale=sm_scale,
            Grid=gd,            # grid num, through compiling
            h_attn=h_attn,      # grid num, through compiling
            bsz=b,           # grid num, through compiling
            C=C, 
            BLK=BLK
        )
        
        
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name, x, km, BLKQ, BLKK,
            sm_scale, tensor_layout, q_or_k
        )
        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "km@OPTIONAL": km,
        }
        out_int8 = helper.create_variable_for_type_inference(dtype=paddle.int8)
        out_scale = helper.create_variable_for_type_inference(dtype=paddle.float32)
        
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "BLKQ": BLKQ,
                "BLKK": BLKK,
                "sm_scale": sm_scale,
                "tensor_layout": tensor_layout,
                "q_or_k": q_or_k
            },
            outputs={"output_tensor": out_int8, "scale_tensor": out_scale}
        )
        return out_int8, out_scale


@paddle_use_triton(
    key=["1"]
)
def sageattn_attn_fwd_casual_false_kernel(
            Q, K, V, Q_scale, K_scale, Out, Lse, 
            stride_qz, stride_qh, stride_qn,
            stride_kz, stride_kh, stride_kn,  
            stride_vz, stride_vh, stride_vn,  
            stride_oz, stride_oh, stride_on,  
            qo_len, kv_len, BSZ,
            H_: tl.constexpr, 
            num_kv_groups: tl.constexpr,
            HEAD_DIM: tl.constexpr,  
            BLOCK_M: tl.constexpr,  
            BLOCK_N: tl.constexpr,  
            STAGE: tl.constexpr,
            RETURN_LSE: tl.constexpr,):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H_ + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H_ // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    
    # fused zone 
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)   
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    # zone end

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H_ + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))
        

def sageattn_forward_casual_false(q, k, v, 
                                  q_scale, k_scale, 
                                  output_dtype="float16",
                                  tensor_layout="HND", 
                                  return_lse=False):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1
    
    assert output_dtype in ["float16", "bfloat16"]
    
    Out = paddle.empty(q.shape, dtype=output_dtype)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_qz, stride_qh, stride_qn = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
        stride_kz, stride_kh, stride_kn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_vz, stride_vh, stride_vn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_oz, stride_oh, stride_on = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        
        stride_qz, stride_qh, stride_qn = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
        stride_kz, stride_kh, stride_kn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_vz, stride_vh, stride_vn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_oz, stride_oh, stride_on = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    BSZ = b
    
    prepare_attr_for_triton_kernel = """
    paddle::DataType output_t;
    if (output_dtype == std::string("float16")) {
        output_t = paddle::DataType::FLOAT16;
    } else {
        output_t = paddle::DataType::BFLOAT16;
    }
    
    auto out_tensor = paddle::empty(q.shape(), output_t, q.place());
    auto q_strides = q.strides();
    auto k_strides = k.strides();
    auto v_strides = v.strides();
    auto o_strides = out_tensor.strides();
    
    int b, h_qo, qo_len, head_dim;
    int kv_len, h_kv;
    
    int stride_qz, stride_qh, stride_qn;
    int stride_kz, stride_kh, stride_kn;
    int stride_vz, stride_vh, stride_vn;
    int stride_oz, stride_oh, stride_on;
    
    if (tensor_layout == "HND") {
        b = q.shape()[0];
        h_qo = q.shape()[1];
        qo_len = q.shape()[2];
        head_dim = q.shape()[3];
        
        h_kv = k.shape()[1];
        kv_len = k.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[1];
        stride_qn = q_strides[2];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[1];
        stride_kn = k_strides[2];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[1];
        stride_vn = v_strides[2];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[1];
        stride_on = o_strides[2];
    } else if (tensor_layout == "NHD") {
        b = q.shape()[0];
        qo_len = q.shape()[1];   // reverse
        h_qo = q.shape()[2];
        head_dim = q.shape()[3];
        
        kv_len = k.shape()[1];   // reverse
        h_kv = k.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[2];       // reverse
        stride_qn = q_strides[1];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[2];       // reverse
        stride_kn = k_strides[1];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[2];       // reverse
        stride_vn = v_strides[1];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[2];       // reverse
        stride_on = o_strides[1];
    } else {
        throw std::runtime_error("Unsupported tensor layout");
    }
    
    int BSZ = b;
"""

    op_name = "triton_sageattn_attn_fwd_casual_false"
    op_name += get_dtype_str(q.dtype)
    op_name += f"_{BLOCK_M}_{BLOCK_N}_BSZ{BSZ}_seq{qo_len}_h{h_qo}_head{HEAD_DIM_K}"
    
    sageattn_attn_fwd_casual_false_config = []
    if head_dim == 64:
        sageattn_attn_fwd_casual_false_config.append({
            "num_warps": 4,
            "num_stages": 3
        })
    else:
        sageattn_attn_fwd_casual_false_config.append({
            "num_warps": 8,
            "num_stages": 4
        })
    
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        if return_lse:
            Lse = paddle.empty((b, h_qo, qo_len), dtype=paddle.float32)
        else:
            Lse = paddle.empty((0, 0, 0), dtype=paddle.float32)
        prepare_ptr_for_triton_kernel = """
        paddle::Tensor lse_tensor;
        
        if (return_lse) {
            lse_tensor = paddle::empty({b, h_qo, qo_len}, q.dtype(), q.place());
        } else {
            lse_tensor = paddle::empty({1,1,1}, paddle::DataType::FLOAT32, paddle::CPUPlace());
        }
        
        auto Q = get_tensor_ptr(q);
        auto K = get_tensor_ptr(k);
        auto V = get_tensor_ptr(v);
        auto Q_scale = get_tensor_ptr(q_scale);
        auto K_scale = get_tensor_ptr(k_scale);
        auto Out = get_tensor_ptr(out_tensor);
        auto Lse = get_tensor_ptr(lse_tensor);
    """
        return_tensor_names = "out_tensor, lse_tensor"
        template_used = rendering_common_template(
            sageattn_forward_casual_false, 
            prepare_attr_for_triton_kernel=prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel=prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names
        )
        grid = ("(qo_len+BLOCK_M-1)/BLOCK_M", "H_", "BSZ")
        sageattn_attn_fwd_casual_false_kernel[(op_name, template_used, grid, sageattn_attn_fwd_casual_false_config)](
            Q=q, 
            K=k, 
            V=v, 
            Q_scale=q_scale, 
            K_scale=k_scale, 
            Out=Out, 
            Lse=Lse, 
            stride_qz=stride_qz, 
            stride_qh=stride_qh, 
            stride_qn=stride_qn,
            stride_kz=stride_kz, 
            stride_kh=stride_kz, 
            stride_kn=stride_kn,
            stride_vz=stride_vz, 
            stride_vh=stride_vh, 
            stride_vn=stride_vn,
            stride_oz=stride_oz, 
            stride_oh=stride_oh, 
            stride_on=stride_on,
            qo_len=qo_len,
            kv_len=kv_len, 
            BSZ=BSZ,
            H_=h_qo, 
            num_kv_groups=num_kv_groups,
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BLOCK_M, 
            BLOCK_N=BLOCK_N, 
            STAGE=stage,
            RETURN_LSE=1 if return_lse else 0
        )
        
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name,
            q,k,v, q_scale, k_scale, 
            output_dtype,
            tensor_layout, 
            return_lse
        )

        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "q": q,
            "k": k,
            "v": v,
            "q_scale": q_scale,
            "k_scale": k_scale,
        }
        out_tensor = helper.create_variable_for_type_inference(dtype=Out.dtype)
        out_lse = helper.create_variable_for_type_inference(dtype=Lse.dtype)
        
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "output_type": output_dtype,
                "tensor_layout": tensor_layout,
                "return_lse": 1 if return_lse else 0
            },
            outputs={
                "out_tensor": out_tensor,
                "lse_tensor": out_lse
            }
        )
        
        return out_tensor, out_lse


# ============== sage attention triton API =================
def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, 
                   tensor_layout="HND"):
    q_int8, q_scale = sageattn_quant_per_block_int8(
        q, km=None, BLKQ=BLKQ, BLKK=BLKK, sm_scale=sm_scale, tensor_layout=tensor_layout, q_or_k='q')
    k_int8, k_scale = sageattn_quant_per_block_int8(
        k, km=km, BLKQ=BLKQ, BLKK=BLKK, sm_scale=sm_scale, tensor_layout=tensor_layout, q_or_k='k')
    return q_int8, q_scale, k_int8, k_scale


def sageattn_qk_int8_pv_fp16_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    tensor_layout: str = "HND",
    is_casual: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs
) -> paddle.Tensor:
    dtype = q.dtype
    assert dtype in [paddle.float16, paddle.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert str(q.place) == str(k.place) == str(v.place), f"All tensors must be on the same device. Got q: {q.place}, k: {k.place}, v: {v.place}"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    head_dim_og = q.shape[-1]
    # if not 64 or 128, then fill to 64 or 128
    if head_dim_og < 64:
        q = paddle.nn.functional.pad(q, pad=[0, 64-head_dim_og])
        k = paddle.nn.functional.pad(k, pad=[0, 64-head_dim_og])
        v = paddle.nn.functional.pad(v, pad=[0, 64-head_dim_og])
    elif head_dim_og > 64 and head_dim_og < 128:
        q = paddle.nn.functional.pad(q, pad=[0, 128-head_dim_og])
        k = paddle.nn.functional.pad(k, pad=[0, 128-head_dim_og])
        v = paddle.nn.functional.pad(v, pad=[0, 128-head_dim_og])
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    seq_dim = 1 if tensor_layout == "NHD" else 2
    
    if smooth_k:
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = paddle.matmul(paddle.transpose(q, [0, 2, 1, 3]), paddle.squeeze(paddle.transpose(q, [0, 2, 3, 1]), axis=-1)).astype(paddle.float32)
            else:
                lse_correction = paddle.matmul(q, paddle.squeeze(paddle.transpose(km, [0, 1, 3, 2]), axis=-1)).astype(paddle.float32)
    else:
        km = None
        
    if dtype == paddle.bfloat16 or dtype == paddle.float32:
        v = paddle.cast(v, dtype=paddle.float16)
        
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)
        
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)

    if is_casual:
        pass
    else:
        o, lse = sageattn_forward_casual_false(q_int8, k_int8, v, q_scale, k_scale, output_dtype="float16", tensor_layout=tensor_layout, return_lse=return_lse)
    
    o = o[..., :head_dim_og]
    
    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o