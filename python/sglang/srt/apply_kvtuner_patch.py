"""
SGLang KVTuner Integration - Patch Script

This script patches SGLang to support KVTuner KV cache quantization.
Run this after SGLang installation to enable KVTuner support.
"""

import os
import sys


def patch_server_args():
    """Add KVTuner arguments to server_args.py"""
    
    server_args_path = "python/sglang/srt/server_args.py"
    
    # Read the file
    with open(server_args_path, 'r') as f:
        content = f.read()
    
    # Add KVTuner config fields after forward_hooks
    kvtuner_fields = '''
    # For forward hooks
    forward_hooks: Optional[List[dict[str, Any]]] = None

    # KVTuner KV Cache Quantization
    enable_kvtuner_quant: bool = False
    kvtuner_nbits_key: int = 4
    kvtuner_nbits_value: int = 4
    kvtuner_asym: bool = False
    kvtuner_axis_key: int = 0
    kvtuner_axis_value: int = 0
    kvtuner_q_group_size: int = 64
    kvtuner_residual_length: int = 128
'''
    
    content = content.replace(
        '''    # For forward hooks
    forward_hooks: Optional[List[dict[str, Any]]] = None

    def __post_init__(self):''',
        kvtuner_fields + '''\n    def __post_init__(self):'''
    )
    
    # Add KVTuner CLI arguments (after forward-hooks argument)
    kvtuner_cli = '''
        # KVTuner KV Cache Quantization
        parser.add_argument(
            "--enable-kvtuner-quant",
            action="store_true",
            help="Enable KVTuner KV cache quantization for memory reduction.",
        )
        parser.add_argument(
            "--kvtuner-nbits-key",
            type=int,
            default=ServerArgs.kvtuner_nbits_key,
            choices=[2, 4, 8],
            help="Number of bits for key quantization (2, 4, or 8).",
        )
        parser.add_argument(
            "--kvtuner-nbits-value",
            type=int,
            default=ServerArgs.kvtuner_nbits_value,
            choices=[2, 4, 8],
            help="Number of bits for value quantization (2, 4, or 8).",
        )
        parser.add_argument(
            "--kvtuner-asym",
            action="store_true",
            default=ServerArgs.kvtuner_asym,
            help="Use asymmetric quantization (default is symmetric).",
        )
        parser.add_argument(
            "--kvtuner-axis-key",
            type=int,
            default=ServerArgs.kvtuner_axis_key,
            choices=[0, 1],
            help="Axis for key quantization: 0=per-token, 1=per-channel.",
        )
        parser.add_argument(
            "--kvtuner-axis-value",
            type=int,
            default=ServerArgs.kvtuner_axis_value,
            choices=[0, 1],
            help="Axis for value quantization: 0=per-token, 1=per-channel.",
        )
        parser.add_argument(
            "--kvtuner-q-group-size",
            type=int,
            default=ServerArgs.kvtuner_q_group_size,
            help="Group size for quantization.",
        )
        parser.add_argument(
            "--kvtuner-residual-length",
            type=int,
            default=ServerArgs.kvtuner_residual_length,
            help="Number of recent tokens to keep in full precision.",
        )
'''
    
    # Insert CLI args before "# KVTuner KV Cache Quantization"
    content = content.replace(
        '''        # Debug tensor dumps
        parser.add_argument(
            "--debug-tensor-dump-output-folder",''',
        kvtuner_cli + '''\n        # Debug tensor dumps
        parser.add_argument(
            "--debug-tensor-dump-output-folder",'''
    )
    
    with open(server_args_path, 'w') as f:
        f.write(content)
    
    print(f"Patched {server_args_path}")


def create_init_patch():
    """Create __init__.py patch for quantization module"""
    
    init_path = "python/sglang/srt/layers/quantization/__init__.py"
    
    if not os.path.exists(init_path):
        print(f"Warning: {init_path} not found")
        return
    
    with open(init_path, 'r') as f:
        content = f.read()
    
    # Add KVTuner import if not present
    if "kvtuner_quant" not in content:
        content = content.replace(
            "from .fp8 import FP8Config",
            """from .fp8 import FP8Config
try:
    from .kvtuner_quant import (
        KVTunerQuantConfig,
        KVTunerQuantizationMethod,
        KVTunerVanillaQuantizer,
    )
    KVTUNER_AVAILABLE = True
except ImportError:
    KVTUNER_AVAILABLE = False"""
        )
        
        with open(init_path, 'w') as f:
            f.write(content)
        
        print(f"Patched {init_path}")


def patch_memory_pool():
    """Patch memory_pool.py to import KVTuner pool"""
    
    pool_path = "python/sglang/srt/mem_cache/memory_pool.py"
    
    with open(pool_path, 'r') as f:
        content = f.read()
    
    # Add import for KVTuner pool
    if "kvtuner_kv_pool" not in content:
        # Find a good place to add import
        import_section = """from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_scale_buffer_triton,
)"""
        
        new_import = import_section + '''\n\n# KVTuner integration\ntry:\n    from sglang.srt.mem_cache.kvtuner_kv_pool import KVTunerMHATokenToKVPool\n    KVTUNER_POOL_AVAILABLE = True\nexcept ImportError:\n    KVTUNER_POOL_AVAILABLE = False\n'''
        
        content = content.replace(import_section, new_import)
        
        with open(pool_path, 'w') as f:
            f.write(content)
        
        print(f"Patched {pool_path}")


if __name__ == "__main__":
    os.chdir("/home/ubuntu/.openclaw/workspace/sglang")
    
    print("Applying KVTuner patches to SGLang...")
    
    # Create backup
    os.system("cp python/sglang/srt/server_args.py python/sglang/srt/server_args.py.bak")
    os.system("cp python/sglang/srt/mem_cache/memory_pool.py python/sglang/srt/mem_cache/memory_pool.py.bak")
    
    patch_server_args()
    create_init_patch()
    patch_memory_pool()
    
    print("\nPatches applied successfully!")
    print("\nTo use KVTuner quantization, run SGLang with:")
    print("  --enable-kvtuner-quant \\")
    print("  --kvtuner-nbits-key 4 \\")
    print("  --kvtuner-nbits-value 4 \\")
    print("  --kvtuner-axis-key 0 \\")
    print("  --kvtuner-axis-value 0")
