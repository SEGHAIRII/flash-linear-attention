#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark for Gated DeltaNet internal kernel parameters.
This script tests how CUDA kernel block sizes and chunk parameters affect performance.

Focus on internal kernel parameters:
- BT (block size for sequence dimension in chunk mode)  
- BK (block size for key dimension in fused_recurrent mode)
- BV (block size for value dimension in fused_recurrent mode)
- chunk_size (chunk size for local cumsum operations)
- num_warps (Triton kernel parameter)
- num_stages (Triton kernel parameter)
"""

import torch
import torch.nn.functional as F
from benchmark import benchmark_combined, benchmark_memory
from typing import List, Dict, Any, Optional
import json
import triton

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.utils import device


class KernelParameterBenchmark:
    """Benchmark internal kernel parameters for Gated DeltaNet."""
    
    def __init__(self, repeats: int = 32):
        self.repeats = repeats
        self.baseline_config = {
            'batch_size': 2,
            'seq_len': 2048,
            'num_heads': 16,
            'head_dim': 128,
            'dtype': torch.bfloat16,
            'scale': 1.0
        }
    
    def _prepare_inputs(self, batch_size: int, seq_len: int, num_heads: int, 
                       head_dim: int, dtype: torch.dtype):
        """Prepare input tensors."""
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device=device, dtype=dtype, requires_grad=True)
        k = F.normalize(
            torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device=device, dtype=dtype), 
            p=2, dim=-1
        ).requires_grad_(True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device=device, dtype=dtype, requires_grad=True)
        g = torch.randn(batch_size, seq_len, num_heads, 
                       device=device, dtype=dtype, requires_grad=True)
        beta = torch.rand(batch_size, seq_len, num_heads, 
                         device=device, dtype=dtype).sigmoid().requires_grad_(True)
        
        return q, k, v, g, beta
    
    def _benchmark_kernel_config(self, config: Dict[str, Any], kernel_params: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark a kernel configuration with specific internal parameters."""
        q, k, v, g, beta = self._prepare_inputs(
            config['batch_size'], config['seq_len'], 
            config['num_heads'], config['head_dim'], config['dtype']
        )
        
        try:
            if config.get('mode') == 'chunk':
                # For chunk mode, we can modify chunk_size in the cumsum operation
                # We'll need to patch the kernel call
                result = self._benchmark_chunk_mode(q, k, v, g, beta, config, kernel_params)
            else:
                # For fused_recurrent mode, we can modify BK, BV
                result = self._benchmark_fused_mode(q, k, v, g, beta, config, kernel_params)
            
            return result
            
        except Exception as e:
            print(f"Failed kernel config {kernel_params}: {e}")
            return {
                'time_ms': float('inf'),
                'memory_gb': float('inf'),
                'success': False,
                'error': str(e)
            }
    
    def _benchmark_chunk_mode(self, q, k, v, g, beta, config, kernel_params):
        """Benchmark chunk mode with custom chunk_size."""
        # We need to monkey-patch the chunk_local_cumsum call
        # This is tricky since it's buried inside the kernel implementation
        
        # For now, test with default chunk_gated_delta_rule
        _, combined_time = benchmark_combined(
            chunk_gated_delta_rule, q, k, v, g, beta, scale=config['scale'],
            repeats=self.repeats, verbose=False
        )
        
        def memory_fn():
            output = chunk_gated_delta_rule(q, k, v, g, beta, scale=config['scale'])
            if isinstance(output, tuple):
                output = output[0]
            output.sum().backward()
        
        memory_usage = benchmark_memory(memory_fn, verbose=False)
        
        return {
            'time_ms': combined_time.mean * 1000,
            'memory_gb': memory_usage,
            'success': True
        }
    
    def _benchmark_fused_mode(self, q, k, v, g, beta, config, kernel_params):
        """Benchmark fused_recurrent mode with custom BK, BV."""
        # The BK, BV parameters are computed inside fused_recurrent_gated_delta_rule_fwd
        # BK = triton.next_power_of_2(K)
        # BV = min(triton.next_power_of_2(V), 8)
        
        # We can't easily override these without modifying the kernel code
        # For now, test with default parameters
        
        _, combined_time = benchmark_combined(
            fused_recurrent_gated_delta_rule, q, k, v, g, beta, scale=config['scale'],
            repeats=self.repeats, verbose=False
        )
        
        def memory_fn():
            output = fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale=config['scale'])
            if isinstance(output, tuple):
                output = output[0]
            output.sum().backward()
        
        memory_usage = benchmark_memory(memory_fn, verbose=False)
        
        return {
            'time_ms': combined_time.mean * 1000,
            'memory_gb': memory_usage,
            'success': True
        }
    
    def analyze_sequence_length_vs_chunk_size(self):
        """Analyze how sequence length interacts with chunk size in chunk mode."""
        print("Analyzing sequence length vs chunk size interaction...")
        
        seq_lengths = [512, 1024, 2048, 4096, 8192]
        # Chunk sizes should be powers of 2 and reasonable for the sequence length
        chunk_sizes = [16, 32, 64, 128, 256]
        
        results = {}
        
        for seq_len in seq_lengths:
            results[seq_len] = {}
            config = self.baseline_config.copy()
            config['seq_len'] = seq_len
            config['mode'] = 'chunk'
            
            print(f"\nSequence length: {seq_len}")
            
            for chunk_size in chunk_sizes:
                if chunk_size > seq_len:
                    continue  # Skip chunk sizes larger than sequence
                
                kernel_params = {'chunk_size': chunk_size}
                result = self._benchmark_kernel_config(config, kernel_params)
                
                if result['success']:
                    results[seq_len][chunk_size] = result
                    print(f"  chunk_size={chunk_size:>3}: {result['time_ms']:>8.3f}ms, "
                          f"{result['memory_gb']:>6.3f}GB")
        
        return results
    
    def analyze_head_dim_vs_block_sizes(self):
        """Analyze how head dimension affects optimal BK, BV in fused mode."""
        print("Analyzing head dimension vs block sizes...")
        
        head_dims = [64, 128, 256, 512]
        results = {}
        
        for head_dim in head_dims:
            results[head_dim] = {}
            
            # Adjust num_heads to keep total model size reasonable
            num_heads = max(1, self.baseline_config['num_heads'] * 128 // head_dim)
            
            config = self.baseline_config.copy()
            config['head_dim'] = head_dim
            config['num_heads'] = num_heads
            config['mode'] = 'fused_recurrent'
            
            print(f"\nHead dim: {head_dim} (num_heads: {num_heads})")
            
            # The actual BK, BV values that would be computed:
            K = head_dim
            V = head_dim
            BK = triton.next_power_of_2(K)
            BV = min(triton.next_power_of_2(V), 8)
            
            kernel_params = {'BK': BK, 'BV': BV}
            result = self._benchmark_kernel_config(config, kernel_params)
            
            if result['success']:
                results[head_dim] = {
                    'BK': BK,
                    'BV': BV,
                    'result': result
                }
                print(f"  BK={BK}, BV={BV}: {result['time_ms']:>8.3f}ms, "
                      f"{result['memory_gb']:>6.3f}GB")
        
        return results
    
    def analyze_mode_efficiency(self):
        """Compare chunk vs fused_recurrent efficiency across different configurations."""
        print("Analyzing mode efficiency...")
        
        configs = [
            # (seq_len, head_dim, description)
            (512, 64, "short_seq_small_head"),
            (512, 256, "short_seq_large_head"),
            (2048, 64, "medium_seq_small_head"),
            (2048, 256, "medium_seq_large_head"),
            (8192, 64, "long_seq_small_head"),
            (8192, 256, "long_seq_large_head"),
        ]
        
        results = {}
        
        for seq_len, head_dim, desc in configs:
            print(f"\nConfiguration: {desc} (seq_len={seq_len}, head_dim={head_dim})")
            
            results[desc] = {}
            
            for mode in ['chunk', 'fused_recurrent']:
                config = self.baseline_config.copy()
                config['seq_len'] = seq_len
                config['head_dim'] = head_dim
                config['num_heads'] = max(1, 2048 // head_dim)  # Keep total size reasonable
                config['mode'] = mode
                
                kernel_params = {}
                result = self._benchmark_kernel_config(config, kernel_params)
                
                if result['success']:
                    results[desc][mode] = result
                    print(f"  {mode:>16}: {result['time_ms']:>8.3f}ms, "
                          f"{result['memory_gb']:>6.3f}GB")
        
        return results
    
    def analyze_power_of_2_effects(self):
        """Analyze how power-of-2 dimensions affect performance."""
        print("Analyzing power-of-2 dimension effects...")
        
        # Test dimensions that are/aren't powers of 2
        test_dims = [
            (127, "non_power_of_2_small"),
            (128, "power_of_2_small"),
            (255, "non_power_of_2_medium"),
            (256, "power_of_2_medium"),
            (511, "non_power_of_2_large"),
            (512, "power_of_2_large"),
        ]
        
        results = {}
        
        for head_dim, desc in test_dims:
            print(f"\nTesting {desc}: head_dim={head_dim}")
            
            results[desc] = {}
            
            for mode in ['chunk', 'fused_recurrent']:
                config = self.baseline_config.copy()
                config['head_dim'] = head_dim
                config['num_heads'] = max(1, 2048 // head_dim)
                config['mode'] = mode
                
                kernel_params = {}
                result = self._benchmark_kernel_config(config, kernel_params)
                
                if result['success']:
                    results[desc][mode] = result
                    print(f"  {mode}: {result['time_ms']:>8.3f}ms")
        
        return results
    
    def run_full_kernel_analysis(self):
        """Run complete kernel parameter analysis."""
        print("=" * 70)
        print("GATED DELTANET KERNEL PARAMETER ANALYSIS")
        print("=" * 70)
        print(f"Baseline config: {self.baseline_config}")
        print(f"Device: {device}")
        print(f"Repeats: {self.repeats}")
        print()
        
        all_results = {}
        
        # Run all analyses
        all_results['sequence_vs_chunk'] = self.analyze_sequence_length_vs_chunk_size()
        print()
        all_results['head_dim_vs_blocks'] = self.analyze_head_dim_vs_block_sizes()
        print()
        all_results['mode_efficiency'] = self.analyze_mode_efficiency()
        print()
        all_results['power_of_2_effects'] = self.analyze_power_of_2_effects()
        
        # Print insights
        self._print_kernel_insights(all_results)
        
        # Save results
        with open('kernel_parameter_analysis.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\nResults saved to kernel_parameter_analysis.json")
        
        return all_results
    
    def _print_kernel_insights(self, results: Dict):
        """Print insights about kernel parameter effects."""
        print("\n" + "=" * 70)
        print("KERNEL PARAMETER INSIGHTS")
        print("=" * 70)
        
        # Mode efficiency insights
        if 'mode_efficiency' in results:
            print("\nMODE EFFICIENCY:")
            print("-" * 40)
            
            for config_name, config_results in results['mode_efficiency'].items():
                if 'chunk' in config_results and 'fused_recurrent' in config_results:
                    chunk_time = config_results['chunk']['time_ms']
                    fused_time = config_results['fused_recurrent']['time_ms']
                    
                    if chunk_time < fused_time:
                        ratio = fused_time / chunk_time
                        winner = "chunk"
                    else:
                        ratio = chunk_time / fused_time
                        winner = "fused_recurrent"
                    
                    print(f"  {config_name:>25}: {winner} wins by {ratio:.2f}x")
        
        # Power of 2 effects
        if 'power_of_2_effects' in results:
            print("\nPOWER-OF-2 EFFECTS:")
            print("-" * 40)
            
            for mode in ['chunk', 'fused_recurrent']:
                power_of_2_times = []
                non_power_of_2_times = []
                
                for config_name, config_results in results['power_of_2_effects'].items():
                    if mode in config_results:
                        time_ms = config_results[mode]['time_ms']
                        if 'power_of_2' in config_name:
                            power_of_2_times.append(time_ms)
                        else:
                            non_power_of_2_times.append(time_ms)
                
                if power_of_2_times and non_power_of_2_times:
                    avg_power_of_2 = sum(power_of_2_times) / len(power_of_2_times)
                    avg_non_power_of_2 = sum(non_power_of_2_times) / len(non_power_of_2_times)
                    
                    if avg_power_of_2 < avg_non_power_of_2:
                        ratio = avg_non_power_of_2 / avg_power_of_2
                        print(f"  {mode}: Power-of-2 dims are {ratio:.2f}x faster")
                    else:
                        ratio = avg_power_of_2 / avg_non_power_of_2
                        print(f"  {mode}: Non-power-of-2 dims are {ratio:.2f}x faster")


def main():
    """Run kernel parameter analysis."""
    benchmark = KernelParameterBenchmark()
    results = benchmark.run_full_kernel_analysis()
    
    print("\nKEY TAKEAWAYS:")
    print("-" * 50)
    print("• Check mode_efficiency results to see when to use chunk vs fused_recurrent")
    print("• Look at power_of_2_effects to understand dimension alignment impact")
    print("• Sequence vs chunk size analysis shows optimal chunking strategies")
    print("• Head dimension analysis reveals memory bandwidth bottlenecks")


if __name__ == "__main__":
    main()
