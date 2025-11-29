#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct kernel parameter benchmark for Gated DeltaNet.
This script tests kernel parameters that actually affect performance:

1. BK, BV (block sizes in fused_recurrent mode) - controlled by head dimension
2. Power-of-2 vs non-power-of-2 dimensions  
3. Sequence length scaling between chunk vs fused modes
4. How head dimensions map to actual kernel block sizes
"""

import torch
import torch.nn.functional as F
import triton
import time

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.utils import device


class DirectKernelBenchmark:
    """Direct benchmark of internal kernel parameters."""
    
    def __init__(self, repeats: int = 32):
        self.repeats = repeats
    
    def prepare_test_data(self, B: int, T: int, H: int, D: int, dtype=torch.bfloat16):
        """Prepare test tensors."""
        q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
        k = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), p=2, dim=-1).requires_grad_(True)
        v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
        g = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
        beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
        return q, k, v, g, beta
    
    def benchmark_power_of_2_dimensions(self):
        """Test how power-of-2 vs non-power-of-2 dimensions affect performance."""
        print("Benchmarking power-of-2 dimension effects...")
        print("=" * 60)
        
        B, T, H = 2, 2048, 16
        
        # Test different head dimensions
        test_dimensions = [
            (63, "non_pow2_small"),
            (64, "pow2_small"),
            (96, "non_pow2_medium"),
            (128, "pow2_medium"),
            (192, "non_pow2_large"),
            (256, "pow2_large"),
        ]
        
        results = {}
        
        for D, desc in test_dimensions:
            print(f"Testing {desc} (D={D})...")
            
            q, k, v, g, beta = self.prepare_test_data(B, T, H, D)
            
            # Test both modes
            for mode in ['chunk', 'fused_recurrent']:
                try:
                    if mode == 'chunk':
                        time_ms = self._benchmark_chunk_mode(q, k, v, g, beta)
                    else:
                        time_ms = self._benchmark_fused_mode(q, k, v, g, beta)
                    
                    config_name = f"{desc}_{mode}"
                    results[config_name] = time_ms
                    print(f"  {mode:>16}: {time_ms:>8.3f} ms")
                    
                except Exception as e:
                    print(f"  {mode:>16}: FAILED ({e})")
        
        # Analyze power-of-2 effects
        print(f"\nPower-of-2 Analysis:")
        print("-" * 30)
        
        for mode in ['chunk', 'fused_recurrent']:
            pow2_times = [results[k] for k in results if 'pow2' in k and mode in k and results[k] != float('inf')]
            non_pow2_times = [results[k] for k in results if 'non_pow2' in k and mode in k and results[k] != float('inf')]
            
            if pow2_times and non_pow2_times:
                avg_pow2 = sum(pow2_times) / len(pow2_times)
                avg_non_pow2 = sum(non_pow2_times) / len(non_pow2_times)
                
                if avg_pow2 < avg_non_pow2:
                    speedup = avg_non_pow2 / avg_pow2
                    print(f"{mode:>16}: pow2 {speedup:.2f}x faster ({avg_pow2:.1f} vs {avg_non_pow2:.1f} ms)")
                else:
                    speedup = avg_pow2 / avg_non_pow2
                    print(f"{mode:>16}: non-pow2 {speedup:.2f}x faster ({avg_non_pow2:.1f} vs {avg_pow2:.1f} ms)")
        
        return results
    
    def _benchmark_chunk_mode(self, q, k, v, g, beta, scale=1.0):
        """Benchmark chunk mode."""
        def kernel_call():
            output = chunk_gated_delta_rule(q, k, v, g, beta, scale=scale)
            if isinstance(output, tuple):
                output = output[0]
            loss = output.sum()
            loss.backward(retain_graph=True)
        
        # Warmup
        for _ in range(5):
            kernel_call()
            torch.cuda.synchronize()
        
        # Timing
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.repeats):
            kernel_call()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return (end_time - start_time) * 1000 / self.repeats
    
    def _benchmark_fused_mode(self, q, k, v, g, beta, scale=1.0):
        """Benchmark fused_recurrent mode."""
        def kernel_call():
            output = fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale=scale)
            if isinstance(output, tuple):
                output = output[0]
            loss = output.sum()
            loss.backward(retain_graph=True)
        
        # Warmup
        for _ in range(5):
            kernel_call()
            torch.cuda.synchronize()
        
        # Timing
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.repeats):
            kernel_call()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return (end_time - start_time) * 1000 / self.repeats
    
    def benchmark_sequence_length_scaling(self):
        """Test how sequence length affects both modes."""
        print("\nBenchmarking sequence length scaling...")
        print("=" * 60)
        
        B, H, D = 2, 16, 128
        seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
        
        results = {}
        
        print(f"{'Seq Len':>8} {'Chunk (ms)':>12} {'Fused (ms)':>12} {'Ratio':>8}")
        print("-" * 45)
        
        for T in seq_lengths:
            q, k, v, g, beta = self.prepare_test_data(B, T, H, D)
            
            try:
                chunk_time = self._benchmark_chunk_mode(q, k, v, g, beta)
                fused_time = self._benchmark_fused_mode(q, k, v, g, beta)
                
                ratio = fused_time / chunk_time if chunk_time > 0 else float('inf')
                
                results[T] = {
                    'chunk': chunk_time,
                    'fused': fused_time,
                    'ratio': ratio
                }
                
                print(f"{T:>8} {chunk_time:>12.3f} {fused_time:>12.3f} {ratio:>8.2f}")
                
            except Exception as e:
                print(f"{T:>8} {'ERROR':>12} {'ERROR':>12} {'ERROR':>8}")
        
        # Find crossover point
        crossover_points = []
        for T in seq_lengths:
            if T in results and results[T]['ratio'] > 1.0:
                crossover_points.append(T)
        
        if crossover_points:
            print(f"\nChunk mode becomes faster starting at sequence length: {min(crossover_points)}")
        else:
            print(f"\nFused mode is consistently faster across all tested sequence lengths")
        
        return results
    
    def benchmark_block_size_effects(self):
        """Test how different block sizes would affect performance by varying head dimensions."""
        print("\nBenchmarking effective block size variations...")
        print("=" * 60)
        
        B, T, H = 2, 2048, 16
        
        # Test head dimensions that result in different BK, BV values
        test_configs = [
            (32, "BK32"),    # BK = next_power_of_2(32) = 32
            (64, "BK64"),    # BK = next_power_of_2(64) = 64  
            (128, "BK128"),  # BK = next_power_of_2(128) = 128
            (256, "BK256"),  # BK = next_power_of_2(256) = 256
        ]
        
        results = {}
        
        print(f"{'Config':>10} {'Head_Dim':>10} {'BK':>6} {'BV':>6} {'Chunk (ms)':>12} {'Fused (ms)':>12}")
        print("-" * 70)
        
        for D, config_name in test_configs:
            BK = triton.next_power_of_2(D)
            BV = min(triton.next_power_of_2(D), 8)
            
            q, k, v, g, beta = self.prepare_test_data(B, T, H, D)
            
            try:
                chunk_time = self._benchmark_chunk_mode(q, k, v, g, beta)
                fused_time = self._benchmark_fused_mode(q, k, v, g, beta)
                
                results[config_name] = {
                    'head_dim': D,
                    'BK': BK,
                    'BV': BV,
                    'chunk': chunk_time,
                    'fused': fused_time
                }
                
                print(f"{config_name:>10} {D:>10} {BK:>6} {BV:>6} {chunk_time:>12.3f} {fused_time:>12.3f}")
                
            except Exception as e:
                print(f"{config_name:>10} {D:>10} {BK:>6} {BV:>6} {'ERROR':>12} {'ERROR':>12}")
        
        return results
    
    def run_complete_kernel_analysis(self):
        """Run all kernel parameter benchmarks."""
        print("GATED DELTANET KERNEL PARAMETER ANALYSIS")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Repeats per test: {self.repeats}")
        print()
        
        all_results = {}
        
        # Run benchmarks
        all_results['power_of_2_effects'] = self.benchmark_power_of_2_dimensions()
        all_results['sequence_scaling'] = self.benchmark_sequence_length_scaling()
        all_results['block_size_effects'] = self.benchmark_block_size_effects()
        
        # Summary insights
        print("\n" + "=" * 70)
        print("KEY INSIGHTS")
        print("=" * 70)
        
        # Mode recommendations
        if 'sequence_scaling' in all_results:
            seq_results = all_results['sequence_scaling']
            chunk_wins = sum(1 for r in seq_results.values() if r.get('ratio', 0) > 1.0)
            total_tests = len(seq_results)
            
            if chunk_wins > total_tests / 2:
                print(f"Chunk mode wins in {chunk_wins}/{total_tests} cases - prefer for longer sequences")
            else:
                print(f"Fused mode wins in {total_tests - chunk_wins}/{total_tests} cases - prefer for shorter sequences")
        
        # Power-of-2 recommendations  
        if 'power_of_2_effects' in all_results:
            print("Power-of-2 dimension analysis completed - check detailed output above")
        
        # Block size recommendations
        if 'block_size_effects' in all_results:
            block_results = all_results['block_size_effects']
            if block_results:
                best_fused = min(block_results.items(), key=lambda x: x[1].get('fused', float('inf')))
                print(f"Best block configuration for fused mode: {best_fused[0]} (BK={best_fused[1]['BK']}, BV={best_fused[1]['BV']})")
        
        return all_results


def main():
    """Run the direct kernel parameter benchmark."""
    benchmark = DirectKernelBenchmark()
    results = benchmark.run_complete_kernel_analysis()
    
    print("\nBenchmark completed! Check the output above for detailed results.")
    print("\nKey takeaways:")
    print("• Power-of-2 dimensions may have alignment advantages")
    print("• Mode choice depends on sequence length and hardware characteristics")
    print("• Block sizes (BK, BV) are determined by head dimensions")


if __name__ == "__main__":
    main()
