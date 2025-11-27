#!/usr/bin/env python3
"""
Trace Analysis Tool for REFLEX PIM Simulator
- Runs ramulator with trace files
- Analyzes performance metrics
- Provides detailed energy and throughput analysis
"""

import argparse
import subprocess
import os
import re
import json
from typing import Dict, List, Optional

class TraceAnalyzer:
    def __init__(self, ramulator_dir: str = "./ramulator2"):
        self.ramulator_dir = ramulator_dir
        self.ramulator_binary = os.path.join(ramulator_dir, "ramulator2")
        self.tCK = 0.769  # ns - HBM3 clock period
        
    def create_yaml_config(self, trace_file: str, power_constraint: bool = False) -> str:
        """Create YAML configuration file for ramulator"""
        yaml_content = f"""Frontend:
  impl: PIMLoadStoreTrace
  path: {trace_file}
  clock_ratio: 1

  Translation:
    impl: NoTranslation
    max_addr: 2147483648

MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: {"HBM3_5.2Gbps" if power_constraint else "HBM3_5.2Gbps_NPC"}

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
    plugins:

  AddrMapper:
    impl: HBM3-PIM
"""
        
        yaml_file = os.path.join(self.ramulator_dir, "temp_config.yaml")
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        return yaml_file
    
    def run_ramulator(self, trace_file: str, power_constraint: bool = False) -> Dict:
        """Run ramulator with the trace file and parse output"""
        print(f"Running ramulator with trace: {trace_file}")
        
        # Create YAML config
        yaml_file = self.create_yaml_config(trace_file, power_constraint)
        
        try:
            # Run ramulator
            cmd = f"{self.ramulator_binary} -f {yaml_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Ramulator failed with return code: {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return {}
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            metrics = self.parse_ramulator_output(output_lines)
            
            return metrics
            
        except Exception as e:
            print(f"Error running ramulator: {e}")
            return {}
        finally:
            # Cleanup
            if os.path.exists(yaml_file):
                os.remove(yaml_file)
    
    def parse_ramulator_output(self, output_lines: List[str]) -> Dict:
        """Parse ramulator output and extract metrics"""
        metrics = {
            "cycles": 0,
            "mac_pb": 0,
            "mac_ab": 0, 
            "mac_sb": 0,
            "sfm": 0,
            "mvgb": 0,
            "mvsb": 0,
            "wrgb": 0
        }
        
        for line in output_lines:
            line = line.strip()
            if "memory_system_cycles" in line:
                metrics["cycles"] = int(line.split()[-1])
            elif "total_num_pim_mac_per_bank_requests" in line:
                metrics["mac_pb"] = int(line.split()[-1])
            elif "total_num_pim_mac_all_bank_requests" in line:
                metrics["mac_ab"] = int(line.split()[-1])
            elif "total_num_pim_mac_same_bank_requests" in line:
                metrics["mac_sb"] = int(line.split()[-1])
            elif "total_num_pim_softmax_requests" in line:
                metrics["sfm"] = int(line.split()[-1])
            elif "total_num_pim_move_to_gemv_buffer_requests" in line:
                metrics["mvgb"] = int(line.split()[-1])
            elif "total_num_pim_move_to_softmax_buffer_requests" in line:
                metrics["mvsb"] = int(line.split()[-1])
            elif "total_num_pim_write_to_gemv_buffer_requests" in line:
                metrics["wrgb"] = int(line.split()[-1])
        
        return metrics
    
    def calculate_performance(self, metrics: Dict, seqlen: int) -> Dict:
        """Calculate performance metrics from ramulator output"""
        cycles = metrics.get("cycles", 0)
        execution_time_ms = cycles * self.tCK / 1e6  # Convert to ms
        
        # Calculate throughput
        throughput = seqlen / (execution_time_ms / 1000) if execution_time_ms > 0 else 0
        
        # Calculate total MAC operations
        total_mac = metrics.get("mac_pb", 0) + metrics.get("mac_ab", 0) + metrics.get("mac_sb", 0)
        
        # Calculate energy (simplified model)
        # These are rough estimates based on typical PIM energy consumption
        dram_energy = cycles * 16 * 50  # 16 channels * 50nJ per cycle
        compute_energy = total_mac * 32000  # ~32uJ per MAC operation
        io_energy = (metrics.get("mvgb", 0) + metrics.get("mvsb", 0) + metrics.get("wrgb", 0)) * 1000
        total_energy = dram_energy + compute_energy + io_energy
        
        # Calculate FLOPs (rough estimate for attention)
        # QK: seqlen * seqlen * dhead, Softmax: seqlen * seqlen, V: seqlen * seqlen * dhead
        dhead = 128  # Assuming standard head dimension
        attention_flops = 2 * seqlen * seqlen * dhead + seqlen * seqlen
        total_flops = attention_flops * 32  # 32 layers assumed
        
        return {
            "execution_time_ms": execution_time_ms,
            "throughput": throughput,
            "total_energy": total_energy,
            "dram_energy": dram_energy,
            "compute_energy": compute_energy,
            "io_energy": io_energy,
            "total_flops": total_flops,
            "energy_efficiency": total_flops / total_energy if total_energy > 0 else 0,
            "energy_per_token": total_energy / seqlen if seqlen > 0 else 0
        }
    
    def calculate_pipelined_performance(self, metrics: Dict, seqlen: int, scheduling_mode: str = "sequential") -> Dict:
        """
        Calculate performance metrics with accurate pipelined row-based execution model
        
        Sequential model (MOD2, MOD8): 
        T_execution ≈ 64t·Row_K + 36t + 32t·Row_V
        
        Pipelined model (MOD2S, MOD8S):
        T_execution_block ≈ 64t·Row_K_block + 36t + 32t·Row_V_block
        
        Block-based pipelining:
        Time:  1  2  3  4  5  6  7  8  9 10 11 12 
        K-row1: QK→SFM→V
        K-row2:    QK→SFM→V  
        K-row3:       QK→SFM→V
        K-row4:          QK→SFM→V
        
        Expected speedup: Up to 36t(#blocks - 1) with long sequences
        """
        cycles = metrics.get("cycles", 0)
        original_execution_time_ms = cycles * self.tCK / 1e6
        
        # Extract timing values from actual ramulator results or use estimates
        # These should be calibrated based on your actual measurements
        t_per_cycle = 1.0  # Clock cycles in tCK units
        
        # Derive actual timing from ramulator metrics if available
        total_mac = metrics.get('mac_pb', 0) + metrics.get('mac_ab', 0) + metrics.get('mac_sb', 0)
        sfm_cmds = metrics.get('sfm', 0)
        
        # Calculate rows from actual command patterns and seqlen
        pages_per_row = 32   # From trace generator configuration
        page_size = 128      # tokens per page
        total_pages = seqlen // page_size
        
        # Estimate rows based on command multipliers
        wrgb_cmds = metrics.get('wrgb', 0)
        estimated_rows_from_wrgb = wrgb_cmds // 16 if wrgb_cmds > 0 else 1  # Assume 16 WR_GB per row
        estimated_rows_from_pages = max(1, total_pages // pages_per_row)
        
        # Use the more conservative estimate
        num_rows = min(estimated_rows_from_wrgb, estimated_rows_from_pages) if estimated_rows_from_wrgb > 0 else estimated_rows_from_pages
        
        # Calculate Row_K and Row_V from actual data
        Row_K = num_rows  # K computation rows
        Row_V = num_rows  # V computation rows (same as K for attention)
        
        performance = self.calculate_performance(metrics, seqlen)
        
        print(f"\n Accurate Timing Analysis (Mode: {scheduling_mode})")
        print(f"Sequence length: {seqlen}, Total pages: {total_pages}")
        print(f"Estimated rows: {num_rows} (from WR_GB: {estimated_rows_from_wrgb}, from pages: {estimated_rows_from_pages})")
        print(f"Row_K: {Row_K}, Row_V: {Row_V}")
        
        if scheduling_mode.endswith('S'):  # MOD2S, MOD8S - pipelined
            # Block-based pipelining analysis
            # Each block contains Row_K_block and Row_V_block
            
            # For MOD2S/MOD8S, blocks are typically smaller than total rows
            if scheduling_mode == "MOD2S":
                rows_per_block = max(1, num_rows // 8)  # Assume 8 blocks for MOD2S
            elif scheduling_mode == "MOD8S":
                rows_per_block = max(1, num_rows // 16) # Assume 16 blocks for MOD8S  
            else:
                rows_per_block = max(1, num_rows // 4)  # Generic case
            
            num_blocks = max(1, num_rows // rows_per_block)
            Row_K_block = rows_per_block
            Row_V_block = rows_per_block  # Same as K for attention
            
            print(f"Block configuration: {num_blocks} blocks, {rows_per_block} rows per block")
            print(f"Row_K_block: {Row_K_block}, Row_V_block: {Row_V_block}")
            
            # Sequential execution time (baseline)
            # T_execution ≈ 64t·Row_K + 36t + 32t·Row_V
            T_sequential = 64 * Row_K + 36 + 32 * Row_V
            
            # Pipelined execution time  
            # T_execution_block ≈ 64t·Row_K_block + 36t + 32t·Row_V_block
            T_block = 64 * Row_K_block + 36 + 32 * Row_V_block
            
            # Total pipelined time with overlap
            # First block: full latency
            # Subsequent blocks: overlap by 36t (SFM latency)
            # Maximum speedup: 36t * (num_blocks - 1)
            overlap_per_block = 36  # SFM latency that can be overlapped
            max_possible_speedup = overlap_per_block * (num_blocks - 1)
            
            # Actual pipelined execution
            if num_blocks == 1:
                T_pipelined = T_block
                actual_speedup = 0
            else:
                # Conservative pipelining model
                T_pipelined = T_block + (num_blocks - 1) * max(64 * Row_K_block, 32 * Row_V_block)
                # Alternative: more aggressive overlap
                # T_pipelined = T_block + (num_blocks - 1) * (T_block - overlap_per_block)
                actual_speedup = T_sequential - T_pipelined
             # Convert to actual cycles (need calibration factor)
            # Use actual ramulator cycles as baseline and scale
            cycle_calibration_factor = cycles / T_sequential if T_sequential > 0 else 1.0
            pipelined_cycles = T_pipelined * cycle_calibration_factor
            
            pipelined_execution_time_ms = pipelined_cycles * self.tCK / 1e6
            
            # Calculate overlap efficiency and other metrics
            overlap_efficiency = (actual_speedup / T_sequential * 100) if T_sequential > 0 else 0
            overlap_cycles_saved = max(0, actual_speedup * cycle_calibration_factor)
            
            # Calculate throughput improvement
            throughput_improvement = original_execution_time_ms / pipelined_execution_time_ms if pipelined_execution_time_ms > 0 else 1.0
            
            print(f"Timing Analysis Results:")
            print(f"Sequential formula: T = 64t·{Row_K} + 36t + 32t·{Row_V} = {T_sequential}t")
            print(f"Pipelined formula: T = 64t·{Row_K_block} + 36t + 32t·{Row_V_block} = {T_block}t per block")
            print(f"Total pipelined time: {T_pipelined}t")
            print(f"Theoretical speedup: {actual_speedup}t ({overlap_efficiency:.1f}%)")
            print(f"Max possible speedup: {max_possible_speedup}t")
            print(f"")
            print(f"Ramulator Results:")
            print(f"Sequential cycles: {cycles}")
            print(f"Estimated pipelined cycles: {pipelined_cycles:.0f}")
            print(f"Cycle calibration factor: {cycle_calibration_factor:.3f}")
            print(f"Throughput improvement: {throughput_improvement:.2f}x")
            
            # Update performance metrics
            performance.update({
                "pipelined_execution_time_ms": pipelined_execution_time_ms,
                "pipelined_throughput": seqlen / (pipelined_execution_time_ms / 1000) if pipelined_execution_time_ms > 0 else 0,
                "throughput_improvement": throughput_improvement,
                "overlap_efficiency": overlap_efficiency,
                "num_rows": num_rows,
                "num_blocks": num_blocks,
                "rows_per_block": rows_per_block,
                "estimated_pipelined_cycles": pipelined_cycles,
                "overlap_cycles_saved": overlap_cycles_saved,
                "T_sequential": T_sequential,
                "T_pipelined": T_pipelined,
                "actual_speedup": actual_speedup,
                "max_possible_speedup": max_possible_speedup
            })
            
        else:  # MOD2, MOD8 - sequential
            print(f"\n Sequential Analysis (Mode: {scheduling_mode})")
            print(f"Total rows: {num_rows}, Sequential execution")
            
            # Calculate Row_K and Row_V for sequential case
            Row_K = num_rows
            Row_V = num_rows
            
            # Estimate theoretical cycles for sequential model
            # T_execution ≈ 64t·Row_K + 36t + 32t·Row_V
            T_sequential = 64 * Row_K + 36 + 32 * Row_V
            cycle_calibration_factor = cycles / T_sequential if T_sequential > 0 else 1.0
            
            print(f"Sequential formula: T = 64t·{Row_K} + 36t + 32t·{Row_V} = {T_sequential}t")
            print(f"Ramulator cycles: {cycles}")
            print(f"Cycle calibration factor: {cycle_calibration_factor:.3f}")
            
            performance.update({
                "estimated_sequential_cycles": cycles,
                "T_sequential": T_sequential,
                "num_rows": num_rows,
                "Row_K": Row_K,
                "Row_V": Row_V,
                "scheduling_mode": "sequential",
                "cycle_calibration_factor": cycle_calibration_factor
            })
        
        return performance

    def analyze_row_based_execution(self, metrics: Dict, seqlen: int, scheduling_mode: str):
        """Detailed analysis of row-based execution patterns"""
        print(f"\n🔍 Row-based Execution Analysis (Mode: {scheduling_mode})")
        
        # Command distribution analysis
        total_mac = metrics.get('mac_pb', 0) + metrics.get('mac_ab', 0) + metrics.get('mac_sb', 0)
        sfm_cmds = metrics.get('sfm', 0)
        wrgb_cmds = metrics.get('wrgb', 0)
        mvsb_cmds = metrics.get('mvsb', 0)
        
        # Estimate rows from command patterns
        pages_per_row = 32
        page_size = 128
        total_pages = seqlen // page_size
        estimated_rows = max(1, total_pages // pages_per_row)
        
        # Analyze command-to-row ratios
        if scheduling_mode.endswith('S'):
            expected_wrgb_multiplier = estimated_rows  # Each row generates WR_GB
            expected_sfm_per_row = sfm_cmds // estimated_rows if estimated_rows > 0 else sfm_cmds
            expected_mvsb_per_row = mvsb_cmds // estimated_rows if estimated_rows > 0 else mvsb_cmds
        else:
            expected_wrgb_multiplier = 1  # Single execution
            expected_sfm_per_row = sfm_cmds
            expected_mvsb_per_row = mvsb_cmds
        
        print(f"Estimated rows: {estimated_rows}")
        print(f"WR_GB commands: {wrgb_cmds} (expected multiplier: {expected_wrgb_multiplier}x)")
        print(f"SFM commands per row: {expected_sfm_per_row}")
        print(f"MVSB commands per row: {expected_mvsb_per_row}")
        print(f"Total MAC commands: {total_mac}")
        
        # Row efficiency analysis
        if estimated_rows > 1:
            row_utilization = total_mac / (estimated_rows * (total_mac // estimated_rows)) if total_mac > 0 else 0
            print(f"Row utilization: {row_utilization:.2f}")

    def print_analysis(self, metrics: Dict, performance: Dict, seqlen: int, scheduling_mode: str = "sequential"):
        """Print detailed analysis results with pipelining support"""
        print("\n📝 Ramulator output analysis:")
        print(f"SFM: total_num_pim_softmax_requests: {metrics.get('sfm', 0)}")
        print(f"MVGB: total_num_pim_move_to_gemv_buffer_requests: {metrics.get('mvgb', 0)}")
        print(f"MVSB: total_num_pim_move_to_softmax_buffer_requests: {metrics.get('mvsb', 0)}")
        print(f"WRGB: total_num_pim_write_to_gemv_buffer_requests: {metrics.get('wrgb', 0)}")
        print(f"MAC: total_num_pim_mac_same_bank_requests: {metrics.get('mac_sb', 0)}")
        print(f"MAC: total_num_pim_mac_all_bank_requests: {metrics.get('mac_ab', 0)}")
        print(f"MAC: total_num_pim_mac_per_bank_requests: {metrics.get('mac_pb', 0)}")
        print(f"CYCLES: memory_system_cycles: {metrics.get('cycles', 0)}")
        
        # Row-based execution analysis
        self.analyze_row_based_execution(metrics, seqlen, scheduling_mode)
        
        # Validate timing calculations
        timing_valid = self.validate_timing_calculations(metrics, performance, scheduling_mode)
        
        print("\n Performance Analysis Results:")
        print(f"Sequence Length: {seqlen}")
        print(f"Scheduling Mode: {scheduling_mode}")
        print(f"Total Cycles: {metrics.get('cycles', 0)}")
        print(f"Execution Time: {performance['execution_time_ms']:.2f} ms")
        print(f"Throughput: {performance['throughput']:.2f} tokens/s")
        
        # Show pipelined results if available
        if 'pipelined_execution_time_ms' in performance:
            print(f"Pipelined Execution Time: {performance['pipelined_execution_time_ms']:.2f} ms")
            print(f"Pipelined Throughput: {performance['pipelined_throughput']:.2f} tokens/s")
            print(f"Throughput Improvement: {performance['throughput_improvement']:.2f}x")
        
        print("\n🔧 Command Breakdown:")
        total_mac = metrics.get('mac_pb', 0) + metrics.get('mac_ab', 0) + metrics.get('mac_sb', 0)
        print(f"MAC Commands: {total_mac}")
        print(f"SFM Commands: {metrics.get('sfm', 0)}")
        print(f"MVGB Commands: {metrics.get('mvgb', 0)}")
        print(f"MVSB Commands: {metrics.get('mvsb', 0)}")
        print(f"WRGB Commands: {metrics.get('wrgb', 0)}")
        
        print("\n⚡ Energy Analysis:")
        print(f"Total Energy: {performance['total_energy']:.2f} nJ")
        print(f"DRAM Energy: {performance['dram_energy']:.2f} nJ ({performance['dram_energy']/performance['total_energy']*100:.1f}%)")
        print(f"Compute Energy: {performance['compute_energy']:.2f} nJ ({performance['compute_energy']/performance['total_energy']*100:.1f}%)")
        print(f"IO Energy: {performance['io_energy']:.2f} nJ ({performance['io_energy']/performance['total_energy']*100:.1f}%)")
        print(f"FLOPs: {performance['total_flops']:.0f}")
        print(f"Energy Efficiency: {performance['energy_efficiency']:.2f} FLOP/nJ")
        
        print("\n🎯 Key Performance Indicators:")
        base_throughput = performance['throughput']
        base_latency = performance['execution_time_ms']
        base_energy_per_token = performance['energy_per_token']
        
        if 'pipelined_throughput' in performance:
            print(f"⚡ Sequential Throughput: {base_throughput:.2f} tokens/s")
            print(f"⚡ Pipelined Throughput: {performance['pipelined_throughput']:.2f} tokens/s")
            print(f"⏱️ Sequential Latency: {base_latency:.2f} ms")
            print(f"⏱️ Pipelined Latency: {performance['pipelined_execution_time_ms']:.2f} ms")
            print(f"🚀 Performance Gain: {performance['throughput_improvement']:.2f}x speedup")
            print(f"🔄 Overlap Efficiency: {performance['overlap_efficiency']:.1f}%")
        else:
            print(f"⚡ Throughput: {base_throughput:.2f} tokens/s")
            print(f"⏱️ Latency: {base_latency:.2f} ms")
        
        print(f"🔋 Energy per Token: {base_energy_per_token:.2f} nJ/token")
        
        # Traffic profile (simplified)
        traffic = [
            metrics.get('wrgb', 0) * 32,  # WR_GB traffic
            metrics.get('mvgb', 0) * 32,  # MV_GB traffic  
            total_mac * 32,               # MAC traffic
            metrics.get('mvsb', 0) * 32,  # MV_SB traffic
            metrics.get('sfm', 0) * 32    # SFM traffic
        ]
        print(f"\n Traffic Profile: {traffic}")
        print(f" FLOPs: {performance['total_flops']:.0f}")

    def validate_timing_calculations(self, metrics: Dict, performance: Dict, scheduling_mode: str):
        """
        Validate if the timing calculations are reasonable by comparing with actual ramulator results
        """
        print(f"\n🔍 Timing Validation for {scheduling_mode}")
        
        cycles = metrics.get("cycles", 0)
        total_mac = metrics.get('mac_pb', 0) + metrics.get('mac_ab', 0) + metrics.get('mac_sb', 0)
        sfm_cmds = metrics.get('sfm', 0)
        wrgb_cmds = metrics.get('wrgb', 0)
        
        # Basic sanity checks
        issues = []
        
        # Check if row calculation makes sense
        if 'num_rows' in performance:
            num_rows = performance['num_rows']
            if num_rows < 1:
                issues.append(" Invalid number of rows (< 1)")
            
            # Check WR_GB to row ratio for S modes
            if scheduling_mode.endswith('S'):
                expected_wrgb_ratio = wrgb_cmds / num_rows if num_rows > 0 else 0
                if expected_wrgb_ratio < 8 or expected_wrgb_ratio > 32:
                    issues.append(f"  Unusual WR_GB/row ratio: {expected_wrgb_ratio:.1f} (expected: 8-32)")
        
        # Check if speedup is too optimistic
        if 'throughput_improvement' in performance:
            improvement = performance['throughput_improvement']
            if improvement > 10:
                issues.append(f"  Very high speedup ({improvement:.1f}x) - may be overoptimistic")
            elif improvement < 0.8:
                issues.append(f"  Speedup less than 1x ({improvement:.1f}x) - check calculation")
        
        # Check if timing formula constants are reasonable
        if 'T_sequential' in performance and 'T_pipelined' in performance:
            T_seq = performance['T_sequential']
            T_pipe = performance['T_pipelined']
            if T_pipe > T_seq:
                issues.append(f"❌ Pipelined time ({T_pipe}t) > Sequential time ({T_seq}t)")
        
        # Report validation results
        if not issues:
            print("✅ Timing calculations appear reasonable")
        else:
            print("⚠️  Potential issues found:")
            for issue in issues:
                print(f"   {issue}")
        
        # Provide recommendations for calibration
        if 'cycle_calibration_factor' in performance:
            cal_factor = performance['cycle_calibration_factor']
            if cal_factor < 0.5 or cal_factor > 2.0:
                print(f"📋 Recommendation: Calibration factor ({cal_factor:.3f}) suggests timing constants may need adjustment")
        
        return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Analyze AttAcc trace files with ramulator")
    parser.add_argument("--trace_file", type=str, required=True, 
                       help="Path to trace file to analyze")
    parser.add_argument("--seqlen", type=int, default=21504,
                       help="Sequence length for throughput calculation")
    parser.add_argument("--ramulator_dir", type=str, default="./ramulator2",
                       help="Directory containing ramulator binary")
    parser.add_argument("--power_constraint", action="store_true",
                       help="Enable power constraint mode")
    parser.add_argument("--scheduling_mode", type=str, default="sequential",
                       choices=["MOD2", "MOD8", "MOD2S", "MOD8S", "LSERVE"],
                       help="Scheduling mode for pipelined analysis")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Check if trace file exists
    if not os.path.exists(args.trace_file):
        print(f" Trace file not found: {args.trace_file}")
        return
    
    # Check if ramulator binary exists
    ramulator_binary = os.path.join(args.ramulator_dir, "ramulator2")
    if not os.path.exists(ramulator_binary):
        print(f" Ramulator binary not found: {ramulator_binary}")
        return
    
    print(f" Analyzing trace file: {args.trace_file}")
    print(f" Scheduling mode: {args.scheduling_mode}")
    
    # Create analyzer and run analysis
    analyzer = TraceAnalyzer(args.ramulator_dir)
    metrics = analyzer.run_ramulator(args.trace_file, args.power_constraint)
    
    if not metrics:
        print(" Failed to get metrics from ramulator")
        return
    
    # Calculate performance metrics (with pipelining if applicable)
    if args.scheduling_mode.endswith('S'):
        performance = analyzer.calculate_pipelined_performance(metrics, args.seqlen, args.scheduling_mode)
    else:
        performance = analyzer.calculate_performance(metrics, args.seqlen)
        performance = analyzer.calculate_pipelined_performance(metrics, args.seqlen, args.scheduling_mode)
    
    # Print analysis
    analyzer.print_analysis(metrics, performance, args.seqlen, args.scheduling_mode)
    
    # Save results if output file specified
    if args.output:
        results = {
            "trace_file": args.trace_file,
            "seqlen": args.seqlen,
            "scheduling_mode": args.scheduling_mode,
            "metrics": metrics,
            "performance": performance
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f" Results saved to: {args.output}")

if __name__ == "__main__":
    main()
