__version__ = "0.2.0"

from hugo.load_packed import load_layer_weight
from hugo.qat import (
    BitLinear,
    bake_bitlinear_to_linear,
    convert_to_bitlinear,
    ternary_fake_quant,
)
from hugo.quantize import (
    LayerQuantStats,
    dequantize_weight,
    pack_ternary_2bit,
    quantize_linear_modules,
    should_skip,
    ternarize_weight,
    unpack_ternary_2bit,
)
from hugo.streaming import (
    copy_aux_files,
    is_quantizable,
    process_shard,
    resolve_weight_map,
)

__all__ = [
    # Quantize (PTQ core math)
    "ternarize_weight",
    "dequantize_weight",
    "quantize_linear_modules",
    "pack_ternary_2bit",
    "unpack_ternary_2bit",
    "LayerQuantStats",
    "should_skip",
    # QAT (quantization-aware training)
    "ternary_fake_quant",
    "BitLinear",
    "convert_to_bitlinear",
    "bake_bitlinear_to_linear",
    # Streaming (disk-bounded I/O)
    "resolve_weight_map",
    "copy_aux_files",
    "is_quantizable",
    "process_shard",
    # Packed loading
    "load_layer_weight",
]
