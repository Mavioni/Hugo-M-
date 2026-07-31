__version__ = "0.2.0"

from hugo.load_packed import load_layer_weight
from hugo.pure import (
    LayerQuantStats,
    _absmean_scale,  # noqa: F401 — public API
    active_code_fraction,
    build_shard_integrity_hash,
    codes_are_ternary,
    compute_tensor_sha256,
    dequantize_weight,
    hash_manifest_shard,
    hash_packed_layer,
    merkle_root,
    pack_ternary_2bit,
    quantization_stats,
    should_skip,
    ternarize_is_contractive,
    ternarize_weight,
    unpack_ternary_2bit,
    verify_manifest_integrity,
)
from hugo.qat import (
    BitLinear,
    bake_bitlinear_to_linear,
    convert_to_bitlinear,
    ternary_fake_quant,
)
from hugo.quantize import (
    quantize_linear_modules,
)
from hugo.streaming import (
    copy_aux_files,
    is_quantizable,
    process_shard,
    resolve_weight_map,
)

try:
    from hugo.openmythos import (
        MythosLMWrapper,
        MythosQATConfig,
        MythosQATTrainer,
        load_mythos_checkpoint,
        quantize_mythos,
    )

    _has_mythos = True
except ImportError:
    _has_mythos = False

__all__ = [
    # Pure math (PTQ)
    "ternarize_weight",
    "dequantize_weight",
    "pack_ternary_2bit",
    "unpack_ternary_2bit",
    "LayerQuantStats",
    "should_skip",
    "codes_are_ternary",
    "active_code_fraction",
    "ternarize_is_contractive",
    "quantization_stats",
    "compute_tensor_sha256",
    "hash_packed_layer",
    "merkle_root",
    "hash_manifest_shard",
    "build_shard_integrity_hash",
    "verify_manifest_integrity",
    # PTQ impure (model mutation)
    "quantize_linear_modules",
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
if _has_mythos:
    __all__ += [
        "load_mythos_checkpoint",
        "quantize_mythos",
        "MythosQATTrainer",
        "MythosQATConfig",
        "MythosLMWrapper",
    ]
