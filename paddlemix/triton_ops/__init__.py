# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = []

try:
    from .rms_norm import rms_norm
    from .triton_ops import (
        adaptive_layer_norm,
        fused_adaLN_scale_residual,
        fused_rotary_emb,
        paddle_use_triton,
        split_concat,
        triton_split,
        sageattn_quant_per_block_int8,
        sageattn_forward_casual_false,
        sageattn_qk_int8_pv_fp16_triton
    )
    from .triton_utils import (
        get_dtype_str,
        paddle_custom_op_head_part,
        tune_and_invoke_part,
    )
    from .wint8 import weight_only_int8

    __all__ += [
        "paddle_custom_op_head_part",
        "tune_and_invoke_part",
        "paddle_use_triton",
        "weight_only_int8",
        "adaptive_layer_norm",
        "fused_adaLN_scale_residual",
        "rms_norm",
        "get_dtype_str",
        "fused_rotary_emb",
        "split_concat",
        "triton_split",
        "sageattn_quant_per_block_int8",
        "sageattn_forward_casual_false",
        "sageattn_qk_int8_pv_fp16_triton"
    ]
except:
    pass
