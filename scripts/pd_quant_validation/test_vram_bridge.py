#!/usr/bin/env python3
"""
NIXL VRAM Bridge Test Script

Tests the VRAM bridge functionality for POSIX backend compatibility.
"""

import os
import sys

# Set POSIX backend before importing SGLang
os.environ["SGLANG_DISAGGREGATION_NIXL_BACKEND"] = "POSIX"

import torch
import time


def test_vram_bridge():
    """Test VRAM bridge functionality."""
    print("=" * 60)
    print("NIXL VRAM Bridge Test")
    print("=" * 60)
    
    # Import bridge
    sys.path.insert(0, "/home/ubuntu/.openclaw/agents/coder/sglang-kvtuner-splitwise/python")
    
    from sglang.srt.disaggregation.nixl.vram_bridge import (
        NIXLVRamBridge,
        get_vram_bridge,
        needs_vram_bridge,
    )
    
    # Check if bridge is needed
    print(f"\n1. Checking if VRAM bridge is needed...")
    needed = needs_vram_bridge()
    print(f"   VRAM bridge needed: {needed}")
    
    # Get bridge instance
    print(f"\n2. Creating VRAM bridge instance...")
    bridge = get_vram_bridge(buffer_size_mb=64, num_buffers=2)
    print(f"   Bridge created: {bridge}")
    print(f"   Needs bridge: {bridge.needs_bridge()}")
    
    # Test VRAM to DRAM copy
    print(f"\n3. Testing VRAM → DRAM copy...")
    if torch.cuda.is_available():
        vram_tensor = torch.randn(1000, 100, dtype=torch.bfloat16, device="cuda")
        print(f"   Created VRAM tensor: {vram_tensor.shape}, {vram_tensor.dtype}")
        
        start = time.time()
        dram_tensor = bridge.copy_vram_to_dram(vram_tensor)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"   Copied to DRAM in {elapsed_ms:.2f}ms")
        print(f"   DRAM tensor shape: {dram_tensor.shape}")
        
        # Verify data
        if torch.allclose(vram_tensor.cpu(), dram_tensor):
            print("   ✅ Data integrity verified")
        else:
            print("   ❌ Data mismatch!")
    else:
        print("   ⚠️ CUDA not available, skipping VRAM test")
    
    # Test DRAM to VRAM copy
    print(f"\n4. Testing DRAM → VRAM copy...")
    if torch.cuda.is_available():
        dram_tensor2 = torch.randn(500, 50, dtype=torch.bfloat16, device="cpu")
        dram_tensor2.pin_memory()
        
        start = time.time()
        vram_tensor2 = bridge.copy_dram_to_vram(dram_tensor2)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"   Copied to VRAM in {elapsed_ms:.2f}ms")
        print(f"   VRAM tensor shape: {vram_tensor2.shape}")
        
        # Verify data
        if torch.allclose(vram_tensor2.cpu(), dram_tensor2):
            print("   ✅ Data integrity verified")
        else:
            print("   ❌ Data mismatch!")
    else:
        print("   ⚠️ CUDA not available, skipping VRAM test")
    
    # Get statistics
    print(f"\n5. Bridge statistics:")
    stats = bridge.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("VRAM Bridge Test Complete")
    print("=" * 60)


def test_kvtuner_quantization():
    """Test KVTuner quantization for KV cache."""
    print("\n" + "=" * 60)
    print("KVTuner Quantization Test")
    print("=" * 60)
    
    sys.path.insert(0, "/home/ubuntu/.openclaw/agents/coder/sglang-kvtuner-splitwise/python")
    
    from sglang.srt.disaggregation.kvtuner.compressed_transfer import (
        KVTunerTransferConfig,
        KVTunerQuantizer,
        CompressionLevel,
    )
    
    # Create config
    print(f"\n1. Creating KVTuner config...")
    config = KVTunerTransferConfig(
        nbits_key=4,
        nbits_value=4,
        q_group_size=64,
        enable_compression=True,
        compression_level=CompressionLevel.BALANCED,
    )
    print(f"   Config: nbits_k={config.nbits_key}, nbits_v={config.nbits_value}")
    
    # Create quantizer
    print(f"\n2. Creating quantizer...")
    quantizer = KVTunerQuantizer(config)
    
    # Test quantization
    if torch.cuda.is_available():
        print(f"\n3. Testing quantization...")
        # Simulate KV cache tensor
        key = torch.randn(128, 32, 128, dtype=torch.bfloat16, device="cuda")  # [tokens, heads, dim]
        value = torch.randn(128, 32, 128, dtype=torch.bfloat16, device="cuda")
        
        print(f"   Original key shape: {key.shape}")
        print(f"   Original key size: {key.numel() * 2 / 1024:.2f} KB")
        
        # Quantize
        start = time.time()
        q_key, k_scale, k_zero = quantizer.quantize(key, nbits=4)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"   Quantized in {elapsed_ms:.2f}ms")
        print(f"   Quantized shape: {q_key.shape}")
        print(f"   Scale shape: {k_scale.shape}")
        
        # Calculate compression
        orig_bytes = key.numel() * 2  # BF16 = 2 bytes
        quant_bytes = q_key.numel() + k_scale.numel() * 4  # int8 + float32 scales
        ratio = orig_bytes / quant_bytes
        
        print(f"   Quantized size: {quant_bytes / 1024:.2f} KB")
        print(f"   Compression ratio: {ratio:.2f}x")
        
        # Dequantize and verify
        print(f"\n4. Testing dequantization...")
        dequant_key = quantizer.dequantize(q_key, k_scale, k_zero, key.shape)
        
        # Calculate error
        max_error = (key - dequant_key).abs().max().item()
        mean_error = (key - dequant_key).abs().mean().item()
        
        print(f"   Max error: {max_error:.6f}")
        print(f"   Mean error: {mean_error:.6f}")
        
        if mean_error < 0.1:  # Typical quantization error
            print("   ✅ Quantization accuracy acceptable")
        else:
            print("   ⚠️ High quantization error")
    else:
        print("   ⚠️ CUDA not available, skipping quantization test")
    
    print("\n" + "=" * 60)
    print("KVTuner Quantization Test Complete")
    print("=" * 60)


def test_posix_patch():
    """Test NIXL POSIX patch."""
    print("\n" + "=" * 60)
    print("NIXL POSIX Patch Test")
    print("=" * 60)
    
    sys.path.insert(0, "/home/ubuntu/.openclaw/agents/coder/sglang-kvtuner-splitwise/python")
    
    from sglang.srt.disaggregation.nixl.posix_patch import (
        patch_nixl_backend,
        check_nixl_backend_status,
    )
    
    # Check status
    print(f"\n1. Checking NIXL backend status...")
    status = check_nixl_backend_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Apply patch
    print(f"\n2. Applying NIXL POSIX patch...")
    try:
        patch_nixl_backend()
        print("   ✅ Patch applied successfully")
    except Exception as e:
        print(f"   ❌ Patch failed: {e}")
    
    print("\n" + "=" * 60)
    print("NIXL POSIX Patch Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SGLang KVTuner + NIXL VRAM Bridge Test Suite")
    print("=" * 60)
    
    test_vram_bridge()
    test_kvtuner_quantization()
    test_posix_patch()
    
    print("\n✅ All tests completed!")#!/usr/bin/env python3
"""
NIXL VRAM Bridge Test Script

Tests the VRAM bridge functionality for POSIX backend compatibility.
"""

import os
import sys

# Set POSIX backend before importing SGLang
os.environ["SGLANG_DISAGGREGATION_NIXL_BACKEND"] = "POSIX"

import torch
import time


def test_vram_bridge():
    """Test VRAM bridge functionality."""
    print("=" * 60)
    print("NIXL VRAM Bridge Test")
    print("=" * 60)
    
    # Import bridge
    sys.path.insert(0, "/home/ubuntu/.openclaw/agents/coder/sglang-kvtuner-splitwise/python")
    
    from sglang.srt.disaggregation.nixl.vram_bridge import (
        NIXLVRamBridge,
        get_vram_bridge,
        needs_vram_bridge,
    )
    
    # Check if bridge is needed
    print(f"\n1. Checking if VRAM bridge is needed...")
    needed = needs_vram_bridge()
    print(f"   VRAM bridge needed: {needed}")
    
    # Get bridge instance
    print(f"\n2. Creating VRAM bridge instance...")
    bridge = get_vram_bridge(buffer_size_mb=64, num_buffers=2)
    print(f"   Bridge created: {bridge}")
    print(f"   Needs bridge: {bridge.needs_bridge()}")
    
    # Test VRAM to DRAM copy
    print(f"\n3. Testing VRAM → DRAM copy...")
    if torch.cuda.is_available():
        vram_tensor = torch.randn(1000, 100, dtype=torch.bfloat16, device="cuda")
        print(f"   Created VRAM tensor: {vram_tensor.shape}, {vram_tensor.dtype}")
        
        start = time.time()
        dram_tensor = bridge.copy_vram_to_dram(vram_tensor)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"   Copied to DRAM in {elapsed_ms:.2f}ms")
        print(f"   DRAM tensor shape: {dram_tensor.shape}")
        
        # Verify data
        if torch.allclose(vram_tensor.cpu(), dram_tensor):
            print("   ✅ Data integrity verified")
        else:
            print("   ❌ Data mismatch!")
    else:
        print("   ⚠️ CUDA not available, skipping VRAM test")
    
    # Get statistics
    print(f"\n4. Bridge statistics:")
    stats = bridge.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("VRAM Bridge Test Complete")
    print("=" * 60)
    return True


def test_kvtuner_quantization():
    """Test KVTuner quantization for KV cache."""
    print("\n" + "=" * 60)
    print("KVTuner Quantization Test")
    print("=" * 60)
    
    sys.path.insert(0, "/home/ubuntu/.openclaw/agents/coder/sglang-kvtuner-splitwise/python")
    
    from sglang.srt.disaggregation.kvtuner.compressed_transfer import (
        KVTunerTransferConfig,
        KVTunerQuantizer,
        CompressionLevel,
    )
    
    # Create config
    print(f"\n1. Creating KVTuner config...")
    config = KVTunerTransferConfig(
        nbits_key=4,
        nbits_value=4,
        q_group_size=64,
        enable_compression=True,
        compression_level=CompressionLevel.BALANCED,
    )
    print(f"   Config: nbits_k={config.nbits_key}, nbits_v={config.nbits_value}")
    
    # Create quantizer
    print(f"\n2. Creating quantizer...")
    quantizer = KVTunerQuantizer(config)
    
    # Test quantization
    if torch.cuda.is_available():
        print(f"\n3. Testing quantization...")
        # Simulate KV cache tensor
        key = torch.randn(128, 32, 128, dtype=torch.bfloat16, device="cuda")
        value = torch.randn(128, 32, 128, dtype=torch.bfloat16, device="cuda")
        
        print(f"   Original key shape: {key.shape}")
        print(f"   Original key size: {key.numel() * 2 / 1024:.2f} KB")
        
        # Quantize
        start = time.time()
        q_key, k_scale, k_zero = quantizer.quantize(key, nbits=4)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"   Quantized in {elapsed_ms:.2f}ms")
        print(f"   Quantized shape: {q_key.shape}")
        print(f"   Scale shape: {k_scale.shape}")
        
        # Calculate compression
        orig_bytes = key.numel() * 2
        quant_bytes = q_key.numel() + k_scale.numel() * 4
        ratio = orig_bytes / quant_bytes
        
        print(f"   Quantized size: {quant_bytes / 1024:.2f} KB")
        print(f"   Compression ratio: {ratio:.2f}x")
        
        # Dequantize and verify
        print(f"\n4. Testing dequantization...")
        dequant_key = quantizer.dequantize(q_key, k_scale, k_zero, key.shape)
        
        max_error = (key - dequant_key).abs().max().item()
        mean_error = (key - dequant_key).abs().mean().item()
        
        print(f"   Max error: {max_error:.6f}")
        print(f"   Mean error: {mean_error:.6f}")
        
        if mean_error < 0.1:
            print("   ✅ Quantization accuracy acceptable")
        else:
            print("   ⚠️ High quantization error")
    else:
        print("   ⚠️ CUDA not available")
    
    return True


if __name__ == "__main__":
    test_vram_bridge()
    test_kvtuner_quantization()
    print("\n✅ All tests completed!")
