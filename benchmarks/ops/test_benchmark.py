#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify the Gated DeltaNet benchmark works correctly.
"""

import torch
import torch.nn.functional as F
from benchmark import benchmark_combined

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.utils import device


def test_gated_deltanet_benchmark():
    """Quick test of the Gated DeltaNet benchmark."""
    print("Testing Gated DeltaNet benchmark...")
    print(f"Device: {device}")
    
    # Test configuration
    B, T, H, D = 2, 1024, 16, 128
    dtype = torch.bfloat16
    scale = 1.0
    
    # Prepare inputs
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = F.normalize(
        torch.randn(B, T, H, D, device=device, dtype=dtype), 
        p=2, dim=-1
    ).requires_grad_(True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, g={g.shape}, beta={beta.shape}")
    
    # Test both modes
    modes = ['chunk', 'fused_recurrent']
    kernels = {
        'chunk': chunk_gated_delta_rule,
        'fused_recurrent': fused_recurrent_gated_delta_rule
    }
    
    for mode in modes:
        print(f"\nTesting {mode} mode...")
        kernel_fn = kernels[mode]
        
        try:
            # Test forward pass
            output = kernel_fn(q, k, v, g, beta, scale=scale)
            if isinstance(output, tuple):
                output = output[0]
            
            print(f"  Forward pass successful, output shape: {output.shape}")
            
            # Test backward pass
            loss = output.sum()
            loss.backward(retain_graph=True)
            print(f"  Backward pass successful")
            
            # Quick benchmark
            _, combined_time = benchmark_combined(
                kernel_fn, q, k, v, g, beta, scale=scale,
                repeats=10, verbose=False
            )
            
            print(f"  Benchmark time: {combined_time.mean * 1000:.3f} ms")
            
        except Exception as e:
            print(f"  Error in {mode} mode: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_gated_deltanet_benchmark()
