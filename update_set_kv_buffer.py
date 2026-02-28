#!/usr/bin/env python3
"""
Update all set_kv_buffer calls to pass forward_batch parameter for KVTuner support.
"""

import os
import re
from pathlib import Path

def update_file(filepath):
    """Update a single file to pass forward_batch to set_kv_buffer calls."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: set_kv_buffer(layer, cache_loc, k, v, layer.k_scale, layer.v_scale)
    # Add forward_batch=forward_batch
    pattern1 = r'(forward_batch\.token_to_kv_pool\.set_kv_buffer\(\s*layer,\s*cache_loc,\s*k,\s*v,\s*layer\.k_scale,\s*layer\.v_scale\s*\))'
    replacement1 = r'forward_batch.token_to_kv_pool.set_kv_buffer(\n                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale,\n                        forward_batch=forward_batch\n                    )'
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: set_kv_buffer(layer, cache_loc, k, v)
    pattern2 = r'(forward_batch\.token_to_kv_pool\.set_kv_buffer\(\s*layer,\s*cache_loc,\s*k,\s*v\s*\))'
    replacement2 = r'forward_batch.token_to_kv_pool.set_kv_buffer(\n                        layer, cache_loc, k, v,\n                        forward_batch=forward_batch\n                    )'
    content = re.sub(pattern2, replacement2, content)
    
    # Pattern 3: set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
    pattern3 = r'(forward_batch\.token_to_kv_pool\.set_kv_buffer\(\s*layer,\s*forward_batch\.out_cache_loc,\s*k,\s*v\s*\))'
    replacement3 = r'forward_batch.token_to_kv_pool.set_kv_buffer(\n                layer, forward_batch.out_cache_loc, k, v,\n                forward_batch=forward_batch\n            )'
    content = re.sub(pattern3, replacement3, content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated: {filepath}")
        return True
    return False

def main():
    base_dir = Path("python/sglang/srt/layers/attention")
    
    files_to_update = [
        "flashattention_backend.py",
        "trtllm_mha_backend.py",
        "xpu_backend.py",
        "wave_backend.py",
        "torch_flex_backend.py",
        "torch_native_backend.py",
        "cutlass_mla_backend.py",
        "flashmla_backend.py",
        "aiter_backend.py",
        "flashinfer_mla_backend.py",
        "double_sparsity_backend.py",
        "dual_chunk_flashattention_backend.py",
    ]
    
    updated_count = 0
    for filename in files_to_update:
        filepath = base_dir / filename
        if filepath.exists():
            if update_file(filepath):
                updated_count += 1
        else:
            print(f"File not found: {filepath}")
    
    print(f"\nUpdated {updated_count} files")

if __name__ == "__main__":
    main()
