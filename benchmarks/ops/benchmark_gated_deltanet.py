#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive benchmark for Gated DeltaNet kernel with different parameters.
This script tests how various parameters affect the runtime performance.

Usage:
    python benchmark_gated_deltanet.py [--mode MODE] [--dtype DTYPE] [--quick]
    
Arguments:
    --mode: 'chunk' or 'fused_recurrent' or 'both' (default: 'both')
    --dtype: 'float16', 'bfloat16', or 'float32' (default: 'bfloat16')
    --quick: Run a quick subset of tests (default: False)
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_memory

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.utils import device


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    batch_size: int
    seq_len: int
    hidden_size: int
    head_dim: int
    num_heads: int
    mode: str
    use_beta: bool = True
    scale: float = 1.0
    dtype: torch.dtype = torch.bfloat16
    
    @property
    def num_kv_heads(self) -> int:
        return self.num_heads
    
    def __str__(self) -> str:
        return (f"{self.name}: B={self.batch_size}, T={self.seq_len}, "
                f"H={self.hidden_size}, D={self.head_dim}, heads={self.num_heads}, "
                f"mode={self.mode}, dtype={self.dtype}")


class GatedDeltaNetBenchmark:
    """Benchmark suite for Gated DeltaNet kernel."""
    
    def __init__(self, repeats: int = 128, warmup: int = 10):
        self.repeats = repeats
        self.warmup = warmup
        self.results = {}
    
    def _prepare_inputs(self, config: BenchmarkConfig) -> Tuple[torch.Tensor, ...]:
        """Prepare input tensors for benchmarking."""
        B, T, H, D = config.batch_size, config.seq_len, config.num_heads, config.head_dim
        
        # Create input tensors
        q = torch.randn(B, T, H, D, device=device, dtype=config.dtype, requires_grad=True)
        k = F.normalize(
            torch.randn(B, T, H, D, device=device, dtype=config.dtype), 
            p=2, dim=-1
        ).requires_grad_(True)
        v = torch.randn(B, T, H, D, device=device, dtype=config.dtype, requires_grad=True)
        
        if config.use_beta:
            beta = torch.rand(B, T, H, device=device, dtype=config.dtype).sigmoid().requires_grad_(True)
        else:
            beta = None
            
        # Gating values for gated delta rule
        g = torch.randn(B, T, H, device=device, dtype=config.dtype, requires_grad=True)
        
        return q, k, v, g, beta
    
    def _get_kernel_fn(self, mode: str):
        """Get the appropriate kernel function."""
        if mode == 'chunk':
            return lambda q, k, v, g, beta, scale: chunk_gated_delta_rule(
                q, k, v, g, beta, scale=scale
            )
        elif mode == 'fused_recurrent':
            return lambda q, k, v, g, beta, scale: fused_recurrent_gated_delta_rule(
                q, k, v, g, beta, scale=scale
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _warmup_kernel(self, config: BenchmarkConfig):
        """Warmup the kernel to avoid cold start effects."""
        q, k, v, g, beta = self._prepare_inputs(config)
        kernel_fn = self._get_kernel_fn(config.mode)
        
        for _ in range(self.warmup):
            try:
                output = kernel_fn(q, k, v, g, beta, config.scale)
                if isinstance(output, tuple):
                    output = output[0]
                output.sum().backward(retain_graph=True)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warmup failed for {config}: {e}")
                return False
        return True
    
    def benchmark_single_config(self, config: BenchmarkConfig) -> Dict:
        """Benchmark a single configuration."""
        print(f"\nBenchmarking: {config}")
        
        # Warmup
        if not self._warmup_kernel(config):
            return {"error": "Warmup failed"}
        
        # Prepare fresh inputs
        q, k, v, g, beta = self._prepare_inputs(config)
        kernel_fn = self._get_kernel_fn(config.mode)
        
        results = {
            "config": config.__dict__,
            "forward_time": None,
            "backward_time": None,
            "combined_time": None,
            "memory_usage": None,
            "error": None
        }
        
        try:
            # Forward pass benchmark
            _, fwd_time = benchmark_forward(
                kernel_fn, q, k, v, g, beta, config.scale,
                repeats=self.repeats,
                desc=f"Forward - {config.name}",
                verbose=False
            )
            results["forward_time"] = fwd_time.mean * 1000  # Convert to ms
            
            # Combined forward+backward benchmark
            _, combined_time = benchmark_combined(
                kernel_fn, q, k, v, g, beta, config.scale,
                repeats=self.repeats,
                desc=f"Combined - {config.name}",
                verbose=False
            )
            results["combined_time"] = combined_time.mean * 1000  # Convert to ms
            results["backward_time"] = results["combined_time"] - results["forward_time"]
            
            # Memory usage
            def memory_fn():
                output = kernel_fn(q, k, v, g, beta, config.scale)
                if isinstance(output, tuple):
                    output = output[0]
                output.sum().backward()
            
            memory_usage = benchmark_memory(
                memory_fn,
                desc=f"Memory - {config.name}",
                verbose=False
            )
            results["memory_usage"] = memory_usage
            
            print(f"  Forward: {results['forward_time']:.4f} ms")
            print(f"  Backward: {results['backward_time']:.4f} ms")
            print(f"  Combined: {results['combined_time']:.4f} ms")
            print(f"  Memory: {results['memory_usage']:.4f} GB")
            
        except Exception as e:
            print(f"  Error: {e}")
            results["error"] = str(e)
        
        return results
    
    def _generate_configs(self, modes: List[str], dtype: torch.dtype, quick: bool = False) -> List[BenchmarkConfig]:
        """Generate benchmark configurations."""
        configs = []
        
        if quick:
            # Quick test configurations
            base_configs = [
                {"batch_size": 2, "seq_len": 1024, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 4, "seq_len": 2048, "hidden_size": 2048, "head_dim": 256, "num_heads": 8},
            ]
        else:
            # Comprehensive test configurations
            base_configs = [
                # Vary sequence length
                {"batch_size": 2, "seq_len": 512, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 2, "seq_len": 1024, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 2, "seq_len": 4096, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                
                # Vary batch size
                {"batch_size": 1, "seq_len": 2048, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 4, "seq_len": 2048, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 8, "seq_len": 2048, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                
                # Vary head dimension
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 64, "num_heads": 32},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 256, "num_heads": 8},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 512, "num_heads": 4},
                
                # Vary hidden size
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 1024, "head_dim": 128, "num_heads": 8},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 4096, "head_dim": 128, "num_heads": 32},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 8192, "head_dim": 128, "num_heads": 64},
                
                # Vary number of heads (keeping hidden_size constant)
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 64, "num_heads": 32},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 128, "num_heads": 16},
                {"batch_size": 2, "seq_len": 2048, "hidden_size": 2048, "head_dim": 256, "num_heads": 8},
                
                # Edge cases
                {"batch_size": 1, "seq_len": 8192, "hidden_size": 4096, "head_dim": 256, "num_heads": 16},
                {"batch_size": 16, "seq_len": 512, "hidden_size": 1024, "head_dim": 64, "num_heads": 16},
            ]
        
        # Generate configs for each mode
        for mode in modes:
            for i, base_config in enumerate(base_configs):
                config = BenchmarkConfig(
                    name=f"{mode}_config_{i+1}",
                    mode=mode,
                    dtype=dtype,
                    **base_config
                )
                configs.append(config)
        
        return configs
    
    def run_benchmark_suite(self, modes: List[str], dtype: torch.dtype, quick: bool = False):
        """Run the complete benchmark suite."""
        print(f"Starting Gated DeltaNet benchmark suite")
        print(f"Modes: {modes}")
        print(f"Data type: {dtype}")
        print(f"Device: {device}")
        print(f"Quick mode: {quick}")
        print(f"Repeats: {self.repeats}")
        print("=" * 70)
        
        configs = self._generate_configs(modes, dtype, quick)
        print(f"Total configurations to test: {len(configs)}")
        
        start_time = time.time()
        
        for i, config in enumerate(configs):
            print(f"\nProgress: {i+1}/{len(configs)}")
            result = self.benchmark_single_config(config)
            self.results[config.name] = result
        
        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.2f} seconds")
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        # Group results by mode
        mode_results = {}
        for name, result in self.results.items():
            if "error" in result and result["error"]:
                continue
            mode = result["config"]["mode"]
            if mode not in mode_results:
                mode_results[mode] = []
            mode_results[mode].append(result)
        
        for mode, results in mode_results.items():
            print(f"\n{mode.upper()} MODE:")
            print("-" * 50)
            
            # Sort by combined time
            results_sorted = sorted(results, key=lambda x: x["combined_time"])
            
            print(f"{'Config':<15} {'Batch':<6} {'SeqLen':<7} {'HeadDim':<8} {'Heads':<6} "
                  f"{'Fwd(ms)':<8} {'Bwd(ms)':<8} {'Total(ms)':<9} {'Mem(GB)':<8}")
            print("-" * 85)
            
            for result in results_sorted:
                config = result["config"]
                print(f"{config['name']:<15} {config['batch_size']:<6} {config['seq_len']:<7} "
                      f"{config['head_dim']:<8} {config['num_heads']:<6} "
                      f"{result['forward_time']:<8.3f} {result['backward_time']:<8.3f} "
                      f"{result['combined_time']:<9.3f} {result['memory_usage']:<8.3f}")
        
        # Performance insights
        print(f"\nPERFORMANCE INSIGHTS:")
        print("-" * 50)
        self._analyze_performance_trends()
    
    def _analyze_performance_trends(self):
        """Analyze and print performance trends."""
        valid_results = [r for r in self.results.values() if not r.get("error")]
        
        if len(valid_results) < 2:
            print("Not enough valid results for analysis")
            return
        
        # Find fastest and slowest configurations
        fastest = min(valid_results, key=lambda x: x["combined_time"])
        slowest = max(valid_results, key=lambda x: x["combined_time"])
        
        print(f"Fastest config: {fastest['config']['name']} - {fastest['combined_time']:.3f} ms")
        print(f"Slowest config: {slowest['config']['name']} - {slowest['combined_time']:.3f} ms")
        print(f"Speed ratio: {slowest['combined_time'] / fastest['combined_time']:.2f}x")
        
        # Memory usage
        min_memory = min(valid_results, key=lambda x: x["memory_usage"])
        max_memory = max(valid_results, key=lambda x: x["memory_usage"])
        
        print(f"Min memory: {min_memory['config']['name']} - {min_memory['memory_usage']:.3f} GB")
        print(f"Max memory: {max_memory['config']['name']} - {max_memory['memory_usage']:.3f} GB")
        
        # Mode comparison
        chunk_results = [r for r in valid_results if r["config"]["mode"] == "chunk"]
        fused_results = [r for r in valid_results if r["config"]["mode"] == "fused_recurrent"]
        
        if chunk_results and fused_results:
            avg_chunk = sum(r["combined_time"] for r in chunk_results) / len(chunk_results)
            avg_fused = sum(r["combined_time"] for r in fused_results) / len(fused_results)
            
            print(f"Average chunk time: {avg_chunk:.3f} ms")
            print(f"Average fused_recurrent time: {avg_fused:.3f} ms")
            if avg_chunk < avg_fused:
                print(f"Chunk mode is {avg_fused/avg_chunk:.2f}x faster on average")
            else:
                print(f"Fused_recurrent mode is {avg_chunk/avg_fused:.2f}x faster on average")
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            # Convert torch.dtype to string for JSON serialization
            serializable_results = {}
            for name, result in self.results.items():
                serializable_result = result.copy()
                if "config" in serializable_result:
                    config = serializable_result["config"].copy()
                    if "dtype" in config:
                        config["dtype"] = str(config["dtype"])
                    serializable_result["config"] = config
                serializable_results[name] = serializable_result
            
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gated DeltaNet kernel")
    parser.add_argument("--mode", choices=["chunk", "fused_recurrent", "both"], 
                        default="both", help="Which mode(s) to benchmark")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], 
                        default="bfloat16", help="Data type to use")
    parser.add_argument("--quick", action="store_true", 
                        help="Run quick benchmark with fewer configurations")
    parser.add_argument("--repeats", type=int, default=128, 
                        help="Number of repeats for each benchmark")
    parser.add_argument("--output", type=str, default="gated_deltanet_benchmark_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # Determine modes to test
    if args.mode == "both":
        modes = ["chunk", "fused_recurrent"]
    else:
        modes = [args.mode]
    
    # Run benchmark
    benchmark = GatedDeltaNetBenchmark(repeats=args.repeats)
    results = benchmark.run_benchmark_suite(modes, dtype, args.quick)
    
    # Save results
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
