#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused benchmark for analyzing specific parameter impacts on Gated DeltaNet performance.
This script systematically varies one parameter at a time to identify performance bottlenecks.
"""

import torch
import torch.nn.functional as F
from benchmark import benchmark_combined, benchmark_memory
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import json

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.utils import device


class ParameterAnalysis:
    """Analyze how individual parameters affect Gated DeltaNet performance."""
    
    def __init__(self, repeats: int = 64):
        self.repeats = repeats
        self.baseline_config = {
            'batch_size': 2,
            'seq_len': 2048,
            'hidden_size': 2048,
            'head_dim': 128,
            'num_heads': 16,
            'mode': 'chunk',
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
    
    def _get_kernel_fn(self, mode: str):
        """Get kernel function."""
        if mode == 'chunk':
            return chunk_gated_delta_rule
        elif mode == 'fused_recurrent':
            return fused_recurrent_gated_delta_rule
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _benchmark_config(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark a single configuration."""
        q, k, v, g, beta = self._prepare_inputs(
            config['batch_size'], config['seq_len'], 
            config['num_heads'], config['head_dim'], config['dtype']
        )
        
        kernel_fn = self._get_kernel_fn(config['mode'])
        
        # Combined time (forward + backward)
        try:
            _, combined_time = benchmark_combined(
                kernel_fn, q, k, v, g, beta, scale=config['scale'],
                repeats=self.repeats, verbose=False
            )
            
            # Memory usage
            def memory_fn():
                output = kernel_fn(q, k, v, g, beta, scale=config['scale'])
                if isinstance(output, tuple):
                    output = output[0]
                output.sum().backward()
            
            memory_usage = benchmark_memory(memory_fn, verbose=False)
            
            return {
                'time_ms': combined_time.mean * 1000,
                'memory_gb': memory_usage,
                'success': True
            }
        except Exception as e:
            print(f"Failed config {config}: {e}")
            return {
                'time_ms': float('inf'),
                'memory_gb': float('inf'),
                'success': False,
                'error': str(e)
            }
    
    def analyze_sequence_length(self, seq_lengths: List[int] = None, modes: List[str] = None):
        """Analyze impact of sequence length."""
        if seq_lengths is None:
            seq_lengths = [512, 1024, 2048, 4096, 8192]
        if modes is None:
            modes = ['chunk', 'fused_recurrent']
        
        print("Analyzing sequence length impact...")
        results = {}
        
        for mode in modes:
            results[mode] = {'seq_lengths': [], 'times': [], 'memories': []}
            
            for seq_len in seq_lengths:
                config = self.baseline_config.copy()
                config['seq_len'] = seq_len
                config['mode'] = mode
                
                result = self._benchmark_config(config)
                if result['success']:
                    results[mode]['seq_lengths'].append(seq_len)
                    results[mode]['times'].append(result['time_ms'])
                    results[mode]['memories'].append(result['memory_gb'])
                    
                    print(f"  {mode} | seq_len={seq_len:>5} | "
                          f"time={result['time_ms']:>8.3f}ms | "
                          f"memory={result['memory_gb']:>6.3f}GB")
        
        return results
    
    def analyze_head_dimension(self, head_dims: List[int] = None, modes: List[str] = None):
        """Analyze impact of head dimension."""
        if head_dims is None:
            head_dims = [64, 128, 256, 512]
        if modes is None:
            modes = ['chunk', 'fused_recurrent']
        
        print("Analyzing head dimension impact...")
        results = {}
        
        for mode in modes:
            results[mode] = {'head_dims': [], 'times': [], 'memories': []}
            
            for head_dim in head_dims:
                # Keep hidden_size constant, adjust num_heads
                num_heads = self.baseline_config['hidden_size'] // head_dim
                if num_heads < 1:
                    continue
                    
                config = self.baseline_config.copy()
                config['head_dim'] = head_dim
                config['num_heads'] = num_heads
                config['mode'] = mode
                
                result = self._benchmark_config(config)
                if result['success']:
                    results[mode]['head_dims'].append(head_dim)
                    results[mode]['times'].append(result['time_ms'])
                    results[mode]['memories'].append(result['memory_gb'])
                    
                    print(f"  {mode} | head_dim={head_dim:>3} (heads={num_heads:>2}) | "
                          f"time={result['time_ms']:>8.3f}ms | "
                          f"memory={result['memory_gb']:>6.3f}GB")
        
        return results
    
    def analyze_batch_size(self, batch_sizes: List[int] = None, modes: List[str] = None):
        """Analyze impact of batch size."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        if modes is None:
            modes = ['chunk', 'fused_recurrent']
        
        print("Analyzing batch size impact...")
        results = {}
        
        for mode in modes:
            results[mode] = {'batch_sizes': [], 'times': [], 'memories': []}
            
            for batch_size in batch_sizes:
                config = self.baseline_config.copy()
                config['batch_size'] = batch_size
                config['mode'] = mode
                
                result = self._benchmark_config(config)
                if result['success']:
                    results[mode]['batch_sizes'].append(batch_size)
                    results[mode]['times'].append(result['time_ms'])
                    results[mode]['memories'].append(result['memory_gb'])
                    
                    print(f"  {mode} | batch_size={batch_size:>2} | "
                          f"time={result['time_ms']:>8.3f}ms | "
                          f"memory={result['memory_gb']:>6.3f}GB")
        
        return results
    
    def analyze_hidden_size(self, hidden_sizes: List[int] = None, modes: List[str] = None):
        """Analyze impact of hidden size."""
        if hidden_sizes is None:
            hidden_sizes = [1024, 2048, 4096, 8192]
        if modes is None:
            modes = ['chunk', 'fused_recurrent']
        
        print("Analyzing hidden size impact...")
        results = {}
        
        for mode in modes:
            results[mode] = {'hidden_sizes': [], 'times': [], 'memories': []}
            
            for hidden_size in hidden_sizes:
                # Keep head_dim constant, adjust num_heads
                num_heads = hidden_size // self.baseline_config['head_dim']
                
                config = self.baseline_config.copy()
                config['hidden_size'] = hidden_size
                config['num_heads'] = num_heads
                config['mode'] = mode
                
                result = self._benchmark_config(config)
                if result['success']:
                    results[mode]['hidden_sizes'].append(hidden_size)
                    results[mode]['times'].append(result['time_ms'])
                    results[mode]['memories'].append(result['memory_gb'])
                    
                    print(f"  {mode} | hidden_size={hidden_size:>4} (heads={num_heads:>2}) | "
                          f"time={result['time_ms']:>8.3f}ms | "
                          f"memory={result['memory_gb']:>6.3f}GB")
        
        return results
    
    def run_full_analysis(self, save_plots: bool = True):
        """Run complete parameter analysis."""
        print("=" * 70)
        print("GATED DELTANET PARAMETER ANALYSIS")
        print("=" * 70)
        print(f"Baseline config: {self.baseline_config}")
        print(f"Device: {device}")
        print(f"Repeats: {self.repeats}")
        print()
        
        all_results = {}
        
        # Run all analyses
        all_results['sequence_length'] = self.analyze_sequence_length()
        print()
        all_results['head_dimension'] = self.analyze_head_dimension()
        print()
        all_results['batch_size'] = self.analyze_batch_size()
        print()
        all_results['hidden_size'] = self.analyze_hidden_size()
        
        # Print summary
        self._print_analysis_summary(all_results)
        
        # Save results
        with open('parameter_analysis_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\nResults saved to parameter_analysis_results.json")
        
        # Create plots if requested
        if save_plots:
            self._create_plots(all_results)
        
        return all_results
    
    def _print_analysis_summary(self, results: Dict):
        """Print analysis summary."""
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        for analysis_name, analysis_results in results.items():
            print(f"\n{analysis_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            for mode in ['chunk', 'fused_recurrent']:
                if mode in analysis_results and analysis_results[mode]['times']:
                    times = analysis_results[mode]['times']
                    min_time = min(times)
                    max_time = max(times)
                    
                    print(f"  {mode:>16}: {min_time:>8.3f} - {max_time:>8.3f} ms "
                          f"(ratio: {max_time/min_time:.2f}x)")
                    
                    # Find best and worst configurations
                    param_name = list(analysis_results[mode].keys())[0]  # First key is the parameter
                    param_values = analysis_results[mode][param_name]
                    
                    min_idx = times.index(min_time)
                    max_idx = times.index(max_time)
                    
                    print(f"                    Best:  {param_name}={param_values[min_idx]}")
                    print(f"                    Worst: {param_name}={param_values[max_idx]}")
    
    def _create_plots(self, results: Dict):
        """Create performance visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Gated DeltaNet Parameter Analysis', fontsize=16)
            
            analyses = [
                ('sequence_length', 'seq_lengths', 'Sequence Length', axes[0, 0]),
                ('head_dimension', 'head_dims', 'Head Dimension', axes[0, 1]),
                ('batch_size', 'batch_sizes', 'Batch Size', axes[1, 0]),
                ('hidden_size', 'hidden_sizes', 'Hidden Size', axes[1, 1])
            ]
            
            for analysis_name, param_key, param_label, ax in analyses:
                if analysis_name in results:
                    for mode in ['chunk', 'fused_recurrent']:
                        if mode in results[analysis_name]:
                            data = results[analysis_name][mode]
                            if data['times']:
                                ax.plot(data[param_key], data['times'], 
                                       marker='o', label=f'{mode} mode')
                
                ax.set_xlabel(param_label)
                ax.set_ylabel('Time (ms)')
                ax.set_title(f'Impact of {param_label}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if analysis_name in ['sequence_length', 'hidden_size']:
                    ax.set_xscale('log', base=2)
                    ax.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig('gated_deltanet_parameter_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Plots saved to gated_deltanet_parameter_analysis.png")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error creating plots: {e}")


def main():
    """Run parameter analysis."""
    analyzer = ParameterAnalysis()
    results = analyzer.run_full_analysis()
    
    # Quick performance insights
    print("\nQUICK INSIGHTS:")
    print("-" * 50)
    
    # Compare modes
    seq_results = results.get('sequence_length', {})
    if 'chunk' in seq_results and 'fused_recurrent' in seq_results:
        chunk_times = seq_results['chunk']['times']
        fused_times = seq_results['fused_recurrent']['times']
        
        if chunk_times and fused_times:
            # Compare on longest sequence
            chunk_long = chunk_times[-1] if chunk_times else float('inf')
            fused_long = fused_times[-1] if fused_times else float('inf')
            
            if chunk_long < fused_long:
                print(f"• Chunk mode is {fused_long/chunk_long:.2f}x faster for long sequences")
            else:
                print(f"• Fused mode is {chunk_long/fused_long:.2f}x faster for long sequences")
    
    # Most impactful parameter
    max_ratios = {}
    for analysis_name, analysis_results in results.items():
        for mode in ['chunk', 'fused_recurrent']:
            if mode in analysis_results and analysis_results[mode]['times']:
                times = analysis_results[mode]['times']
                if len(times) > 1:
                    ratio = max(times) / min(times)
                    max_ratios[f"{analysis_name}_{mode}"] = ratio
    
    if max_ratios:
        worst_param = max(max_ratios.items(), key=lambda x: x[1])
        print(f"• Most impactful parameter: {worst_param[0]} (ratio: {worst_param[1]:.2f}x)")


if __name__ == "__main__":
    main()
