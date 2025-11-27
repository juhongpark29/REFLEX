"""
Page-aware Mobius Trace Generator (clean rewrite)
- Generates Mobius PIM traces with page-based gating capability
- CSV-based (layer, head, time_step) page selection
- MOD-aligned addressing (score/context separately)
- bank / buffer / bg MAC modes
- Optional page-first scheduling for MVGB/SFM
- Detailed mapping/row-activation analysis logs
"""

import argparse
import math
import csv
import json
import os
from typing import List, Optional, Dict, Tuple
from datetime import datetime

# =========================
# Global logs / analyzers
# =========================

mapping_log = []
row_activation_log = []

class MappingAnalyzer:
    """Analyzer for MOD mapping and continuity patterns"""

    def __init__(self, log_file_prefix="mapping_analysis"):
        self.log_file_prefix = log_file_prefix
        self.current_time_step = 0
        self.current_layer = 0
        self.row_access_history = {}   # row_addr(int) -> [time_step,...]
        self.page_to_row_mapping = {}  # page_idx -> row_addr(int)

    def log_page_mapping(self, region_type, page_idx, mod_value, pages_per_row,
                         total_pages, final_addr, time_step, operation):
        # compute row address by clearing column offset
        row_addr = final_addr & ~(HBM_GS['col'] - 1)

        # store quick row/col calc for debug-reporting
        row, col = get_mod_aligned_row_col(page_idx, mod_value, pages_per_row, total_pages)
        group = page_idx % mod_value
        rank = page_idx // mod_value

        mapping_entry = {
            'time_step': time_step,
            'layer': self.current_layer,
            'operation': operation,
            'region_type': region_type,
            'page_idx': page_idx,
            'mod_value': mod_value,
            'pages_per_row': pages_per_row,
            'group': group,
            'rank': rank,
            'calculated_row': row,
            'calculated_col': col,
            'final_addr': hex(final_addr),
            'row_addr': hex(row_addr),
        }
        mapping_log.append(mapping_entry)

        # bookkeep
        self.page_to_row_mapping[page_idx] = row_addr
        self.row_access_history.setdefault(row_addr, []).append(time_step)

        # row activation stream
        row_analyzer.log_row_access(row_addr, time_step, operation, page_idx, mod_value, region_type)

    def check_mod_mapping_correctness(self, mod_value, region_type):
        """Verify if MOD mapping groups pages correctly"""
        expected_groups: Dict[int, List[int]] = {}

        if mod_value == 2:
            expected_groups[0] = [p for p in self.page_to_row_mapping.keys() if p % 2 == 0]
            expected_groups[1] = [p for p in self.page_to_row_mapping.keys() if p % 2 == 1]
        elif mod_value == 8:
            for g in range(8):
                expected_groups[g] = [p for p in self.page_to_row_mapping.keys() if p % 8 == g]
        else:
            # mod 1 => trivial
            expected_groups[0] = list(self.page_to_row_mapping.keys())

        out = []
        for gid, pages in expected_groups.items():
            if len(pages) <= 1:  # nothing to check
                continue
            rows = [self.page_to_row_mapping[p] for p in pages]
            uniq = set(rows)
            out.append({
                'mod_value': mod_value,
                'region_type': region_type,
                'group_id': gid,
                'pages_in_group': pages,
                'row_addresses': [hex(r) for r in rows],
                'unique_rows': len(uniq),
                'is_correct': (len(uniq) == 1),
                'expected_same_row': True
            })
        return out

    def analyze_continuity_breaks(self):
        result = []
        for row_addr, times in self.row_access_history.items():
            if len(times) <= 1:
                continue
            times.sort()
            gaps = []
            for i in range(1, len(times)):
                gap = times[i] - times[i-1]
                if gap > 1:
                    gaps.append({'start_time': times[i-1], 'end_time': times[i], 'gap_size': gap})
            if gaps:
                result.append({
                    'row_addr': hex(row_addr),
                    'total_accesses': len(times),
                    'access_times': times,
                    'gaps': gaps,
                    'continuity_broken': True
                })
        return result

    def save_logs(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        mapping_file = f"{self.log_file_prefix}_mapping_{ts}.json"
        with open(mapping_file, "w") as f:
            json.dump(mapping_log, f, indent=2)

        correctness_mod2 = self.check_mod_mapping_correctness(2, "score_k")
        correctness_mod8 = self.check_mod_mapping_correctness(8, "context_v")
        correctness_file = f"{self.log_file_prefix}_correctness_{ts}.json"
        with open(correctness_file, "w") as f:
            json.dump({'mod2_analysis': correctness_mod2, 'mod8_analysis': correctness_mod8}, f, indent=2)

        continuity = self.analyze_continuity_breaks()
        continuity_file = f"{self.log_file_prefix}_continuity_{ts}.json"
        with open(continuity_file, "w") as f:
            json.dump(continuity, f, indent=2)

        # summary
        summary_file = f"{self.log_file_prefix}_summary_{ts}.txt"
        with open(summary_file, "w") as f:
            f.write("=== MOD Mapping Analysis Summary ===\n\n")
            f.write("1. Mapping Correctness\n")
            f.write(f"   MOD2 groups analyzed: {len(correctness_mod2)}\n")
            mod2_correct = sum(1 for c in correctness_mod2 if c['is_correct'])
            f.write(f"   MOD2 correctly mapped: {mod2_correct}/{len(correctness_mod2)}\n")
            f.write(f"   MOD8 groups analyzed: {len(correctness_mod8)}\n")
            mod8_correct = sum(1 for c in correctness_mod8 if c['is_correct'])
            f.write(f"   MOD8 correctly mapped: {mod8_correct}/{len(correctness_mod8)}\n\n")

            f.write("2. Continuity\n")
            broken = [c for c in continuity if c['continuity_broken']]
            f.write(f"   Rows with broken continuity: {len(broken)}\n")
            f.write(f"   Total unique rows accessed: {len(self.row_access_history)}\n")
        print("Mapping analysis logs saved:")
        print(f"  - {mapping_file}\n  - {correctness_file}\n  - {continuity_file}\n  - {summary_file}")

class RowActivationAnalyzer:
    """Analyzer for row activation patterns and frequency"""

    def __init__(self, log_file_prefix="row_activation"):
        self.log_file_prefix = log_file_prefix
        self.row_access_sequence = []    # sequential accesses
        self.row_activation_count = {}   # row_addr(int) -> count
        self.last_active_row: Optional[int] = None
        self.total_activations = 0
        self.unique_rows_used = set()

    def log_row_access(self, row_addr, time_step, operation, page_idx, mod_value, region_type):
        if isinstance(row_addr, str):
            row_addr = int(row_addr, 16)

        is_activation = (self.last_active_row != row_addr)
        if is_activation:
            self.total_activations += 1
            self.row_activation_count[row_addr] = self.row_activation_count.get(row_addr, 0) + 1
            self.last_active_row = row_addr
        self.unique_rows_used.add(row_addr)

        entry = {
            'time_step': time_step,
            'operation': operation,
            'page_idx': page_idx,
            'mod_value': mod_value,
            'region_type': region_type,
            'row_addr': hex(row_addr),
            'is_activation': is_activation,
            'total_activations_so_far': self.total_activations
        }
        self.row_access_sequence.append(entry)
        row_activation_log.append(entry)

    def analyze_row_activation_efficiency(self):
        total_accesses = len(self.row_access_sequence)
        unique_rows = len(self.unique_rows_used)
        analysis = {
            'total_activations': self.total_activations,
            'unique_rows_used': unique_rows,
            'activation_efficiency': (unique_rows / max(1, self.total_activations)),
            'row_reuse_ratio': ((total_accesses - self.total_activations) / max(1, total_accesses)),
            'activation_distribution': {hex(r): c for r, c in self.row_activation_count.items()}
        }
        avg = self.total_activations / max(1, unique_rows)
        excessive = {hex(r): c for r, c in self.row_activation_count.items() if c > avg * 1.5}
        analysis['excessive_activations'] = excessive
        return analysis

    def compare_mod_efficiency(self, mod_value):
        entries = [e for e in self.row_access_sequence if e['mod_value'] == mod_value]
        if not entries:
            return None
        mod_acts = sum(1 for e in entries if e['is_activation'])
        mod_rows = len(set(int(e['row_addr'], 16) for e in entries))
        return {
            'mod_value': mod_value,
            'total_accesses': len(entries),
            'total_activations': mod_acts,
            'unique_rows': mod_rows,
            'activation_rate': mod_acts / max(1, len(entries)),
            'row_reuse_rate': (len(entries) - mod_acts) / max(1, len(entries)),
            'avg_activations_per_row': mod_acts / max(1, mod_rows)
        }

    def save_activation_analysis(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        seq_file = f"{self.log_file_prefix}_sequence_{ts}.json"
        with open(seq_file, "w") as f:
            json.dump(self.row_access_sequence, f, indent=2)

        analysis = self.analyze_row_activation_efficiency()
        analysis_file = f"{self.log_file_prefix}_analysis_{ts}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

        mod_comparisons = {}
        for mv in [1, 2, 8]:
            a = self.compare_mod_efficiency(mv)
            if a:
                mod_comparisons[f"mod{mv}"] = a
        cmp_file = f"{self.log_file_prefix}_mod_comparison_{ts}.json"
        with open(cmp_file, "w") as f:
            json.dump(mod_comparisons, f, indent=2)

        # summary
        summary = f"{self.log_file_prefix}_summary_{ts}.txt"
        with open(summary, "w") as f:
            f.write("=== Row Activation Analysis Summary ===\n\n")
            f.write(f"Total accesses: {len(self.row_access_sequence)}\n")
            f.write(f"Total activations: {self.total_activations}\n")
            f.write(f"Unique rows used: {len(self.unique_rows_used)}\n")
            f.write(f"Activation efficiency: {analysis['activation_efficiency']:.3f}\n")
            f.write(f"Row reuse ratio: {analysis['row_reuse_ratio']:.3f}\n\n")
            f.write("MOD Comparison:\n")
            for k, v in mod_comparisons.items():
                f.write(f"  {k.upper()} -> accesses={v['total_accesses']}, activations={v['total_activations']}, "
                        f"unique_rows={v['unique_rows']}, activation_rate={v['activation_rate']:.3f}\n")
            if analysis['excessive_activations']:
                f.write("\nRows with excessive activations:\n")
                for r, c in analysis['excessive_activations'].items():
                    f.write(f"  {r}: {c}\n")
        print("Row activation analysis saved:")
        print(f"  - {seq_file}\n  - {analysis_file}\n  - {cmp_file}\n  - {summary}")

# singletons
analyzer = MappingAnalyzer()
row_analyzer = RowActivationAnalyzer()

# =========================
# HBM / system parameters
# =========================

model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16  # bytes per element (FP16 default=2, but overridden by args)

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = 2 ** 14
n_col = 2 ** 5
prefetch_size = 32  # byte (HBM column)
n_mac = 16

HBM_GS = {}
HBM_GS['col'] = prefetch_size
HBM_GS['row'] = n_col * HBM_GS['col']
HBM_GS['ba']  = n_row * HBM_GS['row']
HBM_GS['bg']  = n_bank * HBM_GS['ba']
HBM_GS['rank']= n_bg * HBM_GS['bg']
HBM_GS['pch'] = n_rank * HBM_GS['rank']
HBM_GS['ch']  = n_pch * HBM_GS['pch']
HBM_GS['hbm'] = n_channel * HBM_GS['ch']
HBM_GS['attacc'] = max_n_hbm * HBM_GS['hbm']

# global modes
mac_mode = "buffer"  # 'buffer'|'bank'|'bg'
mod_alignment_score = 2    # for K/Q
mod_alignment_context = 8  # for V

# =========================
# Command containers
# =========================

cmd_score_wrgb = []
cmd_score_mac = []
cmd_score_mvsb = []
cmd_sfm = []
cmd_context_mvgb = []
cmd_context_mac = []
cmd_context_mvsb = []
valid_channels = []

def cmd_list_reset():
    global cmd_score_wrgb, cmd_score_mac, cmd_score_mvsb, cmd_sfm
    global cmd_context_mvgb, cmd_context_mac, cmd_context_mvsb, valid_channels
    cmd_score_wrgb = []
    cmd_score_mac = []
    cmd_score_mvsb = []
    cmd_sfm = []
    cmd_context_mvgb = []
    cmd_context_mac = []
    cmd_context_mvsb = []
    valid_channels = []

# =========================
# Helpers
# =========================

def token_range_for_n_idx(n_idx):
    start = n_idx * n_pch
    end = start + (n_pch - 1)
    return start, end

def token_range_for_k_idx(k_idx):
    start = k_idx * n_pch
    end = start + (n_pch - 1)
    return start, end

def get_pages_per_row_for_region(region_type):
    if region_type in ["score_k", "score_q"]:
        return 4
    elif region_type in ["context_v"]:
        return 32
    return 4

def get_mod_aligned_row_col(page_idx, mod_value, pages_per_row, total_pages):
    group = page_idx % mod_value
    rank  = page_idx // mod_value
    row_in_group = rank // pages_per_row
    col_in_row   = rank %  pages_per_row

    pages_in_group_max = math.ceil(total_pages / max(1, mod_value))
    rows_per_group     = math.ceil(pages_in_group_max / max(1, pages_per_row))

    row = group * rows_per_group + row_in_group
    col = col_in_row
    return row, col

def get_region_base_address(base_addr: int) -> int:
    """
    Clear all row/column bits. Since HBM_GS['row'] is one full DRAM row size,
    clearing lower (row-1) bits zeros both column offset and inside-row offset.
    """
    return base_addr & ~(HBM_GS['row'] - 1)

def get_mod_aligned_address(region_base, page_idx, mod_value, region_type, total_pages):
    pages_per_row = get_pages_per_row_for_region(region_type)
    row, col = get_mod_aligned_row_col(page_idx, mod_value, pages_per_row, total_pages)
    # distribute pages across banks in round-robin
    bank_offset = (page_idx % n_bank) * (HBM_GS['ba'])
    # NOTE: HBM_GS['ba'] is full bank size. If you prefer smaller stride, adjust here.
    aligned_addr = region_base + bank_offset + row * HBM_GS['row'] + col * HBM_GS['col']
    return aligned_addr

def any_token_in_selected_pages(token_start, token_end, page_size, selected_pages):
    if selected_pages is None:
        return True
    if token_start > token_end:
        token_start, token_end = token_end, token_start
    candidates = [token_start, token_end, (token_start + token_end) // 2]
    for t in candidates:
        if t < 0:
            continue
        page = t // page_size
        if page in selected_pages:
            return True
    return False

def pre_filter_valid_indices_by_page(total_steps, get_token_range_func, page_size, selected_pages, csv_data, layer, time_step, head_list, csv_time_steps=None):
    """
    Pre-filter valid indices based on page gating.
    - selected_pages: page-first에서 넘겨준 row 단위 페이지 목록(하드 필터)
    - csv_data: (layer,head,time)별 동적 페이지 (있으면 교집합)
    """
    # no gating at all
    if selected_pages is None and csv_data is None:
        return [{'idx': i, 'page_idx': None, 'time_step': time_step} for i in range(total_steps)]

    valid_indices = []
    # 하드 필터(선택된 row 페이지)
    hard_pages = set(selected_pages) if selected_pages is not None else None

    # 헤드들에서 CSV 기반 페이지 수집
    csv_pages_union = set()
    if csv_data is not None:
        for head in head_list:
            dp = get_dynamic_pages_for_head_timestep(csv_data, layer, head, time_step, [], csv_time_steps)
            if dp:
                csv_pages_union.update(dp)

    # 최종 사용할 페이지 집합 결정
    if hard_pages is not None and len(csv_pages_union) > 0:
        final_pages = hard_pages & csv_pages_union
        if not final_pages:
            return []
    elif hard_pages is not None:
        final_pages = hard_pages
    elif len(csv_pages_union) > 0:
        final_pages = csv_pages_union
    else:
        final_pages = None  # None => 전체 허용

    for idx in range(total_steps):
        t0, t1 = get_token_range_func(idx)
        # 대표 토큰은 시작 토큰으로(이전 코드의 token_idx 근사보다 안전/일관)
        rep_token = t0
        page = rep_token // page_size
        if (final_pages is None) or (page in final_pages):
            valid_indices.append({'idx': idx, 'page_idx': page, 'time_step': time_step})

    return valid_indices

# =========================
# CSV / Page selection
# =========================

def parse_page_list(page_string):
    try:
        return json.loads(page_string.strip())
    except Exception:
        return []

def get_dynamic_pages_for_head_timestep(csv_data, layer, head, time_step, fallback_pages=None, csv_time_steps=None):
    if csv_data is None:
        return fallback_pages
    
    # CSV 데이터의 time_step 범위에 맞게 모듈로 연산
    if csv_time_steps is not None and csv_time_steps > 0:
        time_step_mapped = time_step % csv_time_steps
    else:
        time_step_mapped = time_step
        
    key = (layer, head, time_step_mapped)
    if key in csv_data:
        return csv_data[key]
    # strict match only (no fallback to "any time_step" to reflect true gating)
    return fallback_pages

def load_page_selection_data(csv_path, num_heads=32, num_layers=32, num_time_steps=4):
    page_data = {}
    max_time_step = -1
    if not csv_path or not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found. Using no page gating.")
        return {}, 0

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        row_idx = 0
        for row in reader:
            if not row:
                continue
            time_step = row_idx // num_layers   # 0..3
            layer     = row_idx %  num_layers   # 0..31
            max_time_step = max(max_time_step, time_step)
            for head, cell in enumerate(row):
                if cell.strip():
                    pages = parse_page_list(cell)
                    if pages:
                        page_data[(layer, head, time_step)] = pages
            row_idx += 1

    print(f"Loaded page data for {len(page_data)} (layer, head, time_step) combinations")
    print(f"CSV contains time_steps 0..{max_time_step}")
    return page_data, max_time_step + 1

def num_pages(L, page_size):
    return math.ceil(L / page_size)

def build_selected_pages(mode, L, page_size,
                         include_pages: Optional[List[int]] = None,
                         exclude_pages: Optional[List[int]] = None) -> Optional[List[int]]:
    n = num_pages(L, page_size)
    pages_all = list(range(n))
    if mode == 'all':
        return None
    if mode == 'even':
        return [p for p in pages_all if (p % 2) == 0]
    if mode == 'odd':
        return [p for p in pages_all if (p % 2) == 1]
    if mode == 'first_half':
        return list(range(0, n // 2))
    if mode == 'second_half':
        return list(range(n // 2, n))
    if mode == 'exclude_back_half':
        return list(range(0, n // 2))
    if mode == 'only_list':
        assert include_pages is not None, "include_pages required"
        return sorted([p for p in include_pages if 0 <= p < n])
    if mode == 'exclude_list':
        assert exclude_pages is not None, "exclude_pages required"
        s = set(exclude_pages)
        return [p for p in pages_all if p not in s]
    raise ValueError(f"Unknown mode: {mode}")

# =========================
# Core generator
# =========================

def Attention(L: int, key_addr: int, val_addr: int, itr: int, page_size: int,
              selected_pages: Optional[List[int]], csv_data: Optional[Dict] = None,
              valid_channel: int = n_channel, layer: int = 0, time_step: int = 0,
              page_scheduling: bool = False, head_base: int = 0, csv_time_steps: Optional[int] = None):
    """Generate AttAcc attention commands with page gating"""
    # per-itr containers
    cmd_score_wrgb.append([])
    cmd_score_mac.append([])
    cmd_score_mvsb.append([])
    cmd_sfm.append([])
    cmd_context_mvgb.append([])
    cmd_context_mac.append([])
    cmd_context_mvsb.append([])
    valid_channels.append(valid_channel)

    # Global flag to ensure WR_GB is only generated once per attention call
    wrgb_generated = False

    def score_cpvec(addr_offset: int, L: int):
        nonlocal wrgb_generated
        # For page scheduling mode, apply global deduplication to prevent multiple WR_GB
        if page_scheduling:
            # Only generate WR_GB if not already issued globally - use layer for granularity
            for col_idx in range(math.ceil(dhead / n_mac)):
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + col_idx
                    addr_key = (addr, "WR_GB", layer)  # layer-based deduplication
                    if addr_key not in global_wrgb_issued:
                        cmd_score_wrgb[itr].append(f"PIM_WR_GB 0x{hex(addr)[2:]:0>8}")
                        global_wrgb_issued.add(addr_key)
            wrgb_generated = True
        else:
            # WR_GB should be executed once per attention call, not per layer/head/time_step
            if not wrgb_generated:
                for col_idx in range(math.ceil(dhead / n_mac)):
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch'] + col_idx
                        cmd_score_wrgb[itr].append(f"PIM_WR_GB 0x{hex(addr)[2:]:0>8}")
                wrgb_generated = True

    def score_mac(addr_offset: int, L: int):
        total_pages = math.ceil(L / page_size)
        mod_value = mod_alignment_score

        def token_range_for_n_idx_local(n_idx):
            if mac_mode == "buffer":
                return (n_idx * n_pch, (n_idx + 1) * n_pch - 1)
            elif mac_mode == "bank":
                factor = n_pch * n_rank * n_bg
                return (n_idx * factor, (n_idx + 1) * factor - 1)
            else:  # bg
                factor = n_pch * n_rank * n_bg
                return (n_idx * factor, (n_idx + 1) * factor - 1)

        # Pre-filter valid indices using 사전 게이팅
        if mac_mode == "buffer":
            N_steps = math.ceil(L / n_pch)
        elif mac_mode == "bank":
            N_steps = math.ceil(L / n_pch / n_rank / n_bg)
        else:  # bg
            N_steps = math.ceil(L / n_pch / n_rank / n_bg)

        # Create list of heads for this iteration
        head_list = []
        for ch_off in range(valid_channel):
            head = head_base + ch_off           # 레이어-로컬 기준
            head_list.append(head)

        # Pre-filter valid n_idx values
        valid_n_list = pre_filter_valid_indices_by_page(
            N_steps, token_range_for_n_idx_local, page_size, selected_pages, 
            csv_data, layer, time_step, head_list, csv_time_steps
        )

        # Sort by row address for better locality
        valid_n_enhanced = []
        for entry in valid_n_list:
            n_idx = entry['idx']
            if mac_mode == "buffer":
                token_idx = n_idx * n_pch
            else:  # bank or bg
                token_idx = n_idx * n_pch * n_rank * n_bg
            page_idx = token_idx // page_size
            region_base = get_region_base_address(addr_offset)
            aligned_addr = get_mod_aligned_address(region_base, page_idx, mod_value, "score_k", total_pages)
            row_addr = aligned_addr & ~(HBM_GS['col'] - 1)
            valid_n_enhanced.append({
                'n_idx': n_idx,
                'page_idx': page_idx, 
                'row_addr': row_addr,
                'time_step': entry['time_step']
            })
####row 단위의 trace 생성
        valid_n_enhanced.sort(key=lambda x: x['row_addr'])  # DISABLED to match ORI behavior
        
        # Log filtering results
        filtered = N_steps - len(valid_n_enhanced)
        fr = (filtered / N_steps * 100) if N_steps else 0.0
        print(f"    Score MAC ({mac_mode}): Total steps={N_steps}, Valid steps={len(valid_n_enhanced)}, Filtered={filtered} ({fr:.1f}%)")
        if valid_n_enhanced:
            uniq_rows = len({x['row_addr'] for x in valid_n_enhanced})
            print(f"    Score MAC ({mac_mode}): Row consolidation: {len(valid_n_enhanced)} -> {uniq_rows} rows")

        # Generate commands only for valid indices
        for entry in valid_n_enhanced:
            n_idx = entry['n_idx']
            page_idx = entry['page_idx']
            ts_local = entry['time_step']
            
            cmd_score_mac[itr].append([])
            
            # K dimension iteration
            if mac_mode == "buffer":
                k_range = math.ceil(dhead / n_mac)
                opcode = "PIM_MAC_PB"
            elif mac_mode == "bank":
                k_range = math.ceil(dhead / n_bank / n_mac)
                opcode = "PIM_MAC_AB"
            else:  # bg
                k_range = math.ceil(dhead / n_mac)
                opcode = "PIM_MAC_SB"
                
            for k_idx in range(k_range):
                region_base = get_region_base_address(addr_offset)
                for lch in range(math.ceil(valid_channel)):
                    ch_base = region_base + lch * HBM_GS['ch']
                    aligned_addr = get_mod_aligned_address(ch_base, page_idx, mod_value, "score_k", total_pages)
                    addr = aligned_addr + k_idx * HBM_GS['col']
                    
                    analyzer.log_page_mapping("score_k", page_idx, mod_value,
                                              get_pages_per_row_for_region("score_k"),
                                              total_pages, addr, ts_local,
                                              f"score_mac_{mac_mode}_n{n_idx}_k{k_idx}_ch{lch}")
                    
                    cmd_score_mac[itr][-1].append(f"{opcode} 0x{hex(addr)[2:]:0>8}")

        # MVSB cadence with intelligent deduplication for page scheduling
        m = 16
        for i, _ in enumerate(valid_n_enhanced):
            if (i + 1) % m == 0 or i == len(valid_n_enhanced) - 1:
                cmd_score_mvsb[itr].append([])
                if mac_mode == "buffer":
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch']
                        if page_scheduling:
                            # Use layer + time_step for more granular deduplication
                            addr_key = (addr, "MV_SB_score", layer, time_step)
                            if addr_key not in global_mvsb_issued:
                                cmd_score_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")
                                global_mvsb_issued.add(addr_key)
                        else:
                            cmd_score_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")
                elif mac_mode == "bank":
                    for bg_idx in range(n_bg):
                        for rk in range(n_rank):
                            for lch in range(math.ceil(valid_channel)):
                                bank_addr = (addr_offset + lch * HBM_GS['ch'] +
                                             rk * HBM_GS['rank'] + bg_idx * HBM_GS['bg'])
                                if page_scheduling:
                                    addr_key = (bank_addr, "MV_SB_score", layer, time_step)
                                    if addr_key not in global_mvsb_issued:
                                        cmd_score_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(bank_addr)[2:]:0>8}")
                                        global_mvsb_issued.add(addr_key)
                                else:
                                    cmd_score_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(bank_addr)[2:]:0>8}")
                else:  # bg
                    for rk in range(n_rank):
                        for lch in range(math.ceil(valid_channel)):
                            addr = addr_offset + lch * HBM_GS['ch'] + rk * HBM_GS['rank']
                            if page_scheduling:
                                addr_key = (addr, "MV_SB_score", layer, time_step)
                                if addr_key not in global_mvsb_issued:
                                    cmd_score_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")
                                    global_mvsb_issued.add(addr_key)
                            else:
                                cmd_score_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")
    def context_cpvec(addr_offset: int, L: int):
        steps = math.ceil(L / (n_pch * n_mac))
        
        def token_range_for_col_idx(col_idx):
            factor = n_pch * n_mac
            return (col_idx * factor, (col_idx + 1) * factor - 1)

        # Pre-filter valid col_idx values using 사전 게이팅
        head_list = [head_base]        # 레이어-로컬 기준 하나
        
        valid_col_list = pre_filter_valid_indices_by_page(
            steps, token_range_for_col_idx, page_size, selected_pages, 
            csv_data, layer, time_step, head_list, csv_time_steps
        )

        print(f"    Context MVGB: Total steps={steps}, Valid steps={len(valid_col_list)}, Filtered={steps - len(valid_col_list)}")

        if page_scheduling:
            pages_per_chunk = 8
            # Group valid col_idx by chunks for page scheduling
            chunk_groups = {}
            for entry in valid_col_list:
                col_idx = entry['idx']
                chunk_idx = col_idx // pages_per_chunk
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(entry)
            
            print(f"    Page scheduling enabled: Processing {len(chunk_groups)} chunks")
            
            for chunk_idx in sorted(chunk_groups.keys()):
                entries = chunk_groups[chunk_idx]
                #print(f"      Processing MVGB chunk {chunk_idx}: {len(entries)} valid steps")
                for entry in entries:
                    col_idx = entry['idx']
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch'] + col_idx
                        cmd_context_mvgb[itr].append(f"PIM_MV_GB 0x{hex(addr)[2:]:0>8}")
        else:
            # Process all valid col_idx directly
            for entry in valid_col_list:
                col_idx = entry['idx']
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + col_idx
                    cmd_context_mvgb[itr].append(f"PIM_MV_GB 0x{hex(addr)[2:]:0>8}")

    def context_mac(addr_offset: int, L: int):
        total_pages = math.ceil(L / page_size)
        mod_value = mod_alignment_context

        def token_range_for_k_idx_local(k_idx):
            if mac_mode == "buffer":
                return (k_idx * n_pch, (k_idx + 1) * n_pch - 1)
            elif mac_mode == "bank":
                factor = n_pch * n_rank * n_bg
                return (k_idx * factor, (k_idx + 1) * factor - 1)
            else:  # bg
                factor = n_pch * n_rank * n_bg
                return (k_idx * factor, (k_idx + 1) * factor - 1)

        def process_k_indices(K_steps, per_k_col_factor, mac_opcode):
            for n_idx in range(math.ceil(dhead / per_k_col_factor)):
                # Pre-filter valid k_idx values using 사전 게이팅
                # Use the same head_list as Score MAC for consistent page filtering
                head_list = []
                for ch_off in range(valid_channel):
                    head = head_base + ch_off           # 레이어-로컬 기준
                    head_list.append(head)
                
                valid_k_list = pre_filter_valid_indices_by_page(
                    K_steps, token_range_for_k_idx_local, page_size, selected_pages, 
                    csv_data, layer, time_step, head_list, csv_time_steps
                )

                # Sort by row address for better locality
                valid_k_enhanced = []
                for entry in valid_k_list:
                    k_idx = entry['idx']
                    token_idx = k_idx * n_pch
                    page_idx = token_idx // page_size
                    region_base = get_region_base_address(addr_offset)
                    aligned_addr = get_mod_aligned_address(region_base, page_idx, mod_value, "context_v", total_pages)
                    row_addr = aligned_addr & ~(HBM_GS['col'] - 1)
                    valid_k_enhanced.append({
                        'k_idx': k_idx,
                        'page_idx': page_idx, 
                        'row_addr': row_addr,
                        'time_step': entry['time_step']
                    })
####row 단위의 trace 생성
                valid_k_enhanced.sort(key=lambda x: x['row_addr'])  # DISABLED to match ORI behavior
                
                # Log filtering results
                filtered = K_steps - len(valid_k_enhanced)
                fr = (filtered / K_steps * 100) if K_steps else 0.0
                print(f"    Context MAC ({mac_mode}) n_idx {n_idx}: Total k_steps={K_steps}, Valid k_steps={len(valid_k_enhanced)}, Filtered={filtered} ({fr:.1f}%)")
                if valid_k_enhanced:
                    uniq_rows = len({x['row_addr'] for x in valid_k_enhanced})
                    print(f"    Context MAC ({mac_mode}) n_idx {n_idx}: Row consolidation: {len(valid_k_enhanced)} -> {uniq_rows} rows")

                cmd_context_mac[itr].append([])
                for entry in valid_k_enhanced:
                    k_idx = entry['k_idx']
                    page_idx = entry['page_idx']
                    ts_local = entry['time_step']
                    
                    region_base = get_region_base_address(addr_offset)
                    for lch in range(math.ceil(valid_channel)):
                        ch_base = region_base + lch * HBM_GS['ch']
                        aligned_addr = get_mod_aligned_address(ch_base, page_idx, mod_value, "context_v", total_pages)
                        addr = aligned_addr + n_idx * HBM_GS['col']
                        analyzer.log_page_mapping("context_v", page_idx, mod_value,
                                                  get_pages_per_row_for_region("context_v"),
                                                  total_pages, addr, ts_local,
                                                  f"context_mac_{mac_mode}_n{n_idx}_k{k_idx}_ch{lch}")
                        cmd_context_mac[itr][-1].append(f"{mac_opcode} 0x{hex(addr)[2:]:0>8}")

                # per n_idx MV_SB to keep pipeline shape with intelligent deduplication
                cmd_context_mvsb[itr].append([])
                if mac_mode == "bank":
                    for ba_idx in range(n_bank):
                        for rk in range(n_rank):
                            for lch in range(math.ceil(valid_channel)):
                                bank_addr = (addr_offset + lch * HBM_GS['ch'] +
                                             rk * HBM_GS['rank'] + ba_idx * HBM_GS['ba'])
                                if page_scheduling:
                                    # Use layer + time_step + n_idx for more granular deduplication
                                    addr_key = (bank_addr, "MV_SB_context", layer, time_step, n_idx)
                                    if addr_key not in global_mvsb_issued:
                                        cmd_context_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(bank_addr)[2:]:0>8}")
                                        global_mvsb_issued.add(addr_key)
                                else:
                                    cmd_context_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(bank_addr)[2:]:0>8}")
                elif mac_mode == "bg":
                    for rk in range(n_rank):
                        for lch in range(math.ceil(valid_channel)):
                            addr = addr_offset + lch * HBM_GS['ch'] + rk * HBM_GS['rank']
                            if page_scheduling:
                                addr_key = (addr, "MV_SB_context", layer, time_step, n_idx)
                                if addr_key not in global_mvsb_issued:
                                    cmd_context_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")
                                    global_mvsb_issued.add(addr_key)
                            else:
                                cmd_context_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")
                else:
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch']
                        if page_scheduling:
                            addr_key = (addr, "MV_SB_context", layer, time_step, n_idx)
                            if addr_key not in global_mvsb_issued:
                                cmd_context_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")
                                global_mvsb_issued.add(addr_key)
                        else:
                            cmd_context_mvsb[itr][-1].append(f"PIM_MV_SB 0x{hex(addr)[2:]:0>8}")

        if mac_mode == "buffer":
            K_steps = math.ceil(L / n_pch)             # per-bank
            per_k_col_factor = n_mac
            mac_opcode = "PIM_MAC_PB"
        elif mac_mode == "bank":
            K_steps = math.ceil(L / n_pch / n_rank / n_bg)
            per_k_col_factor = (n_bank * n_mac)
            mac_opcode = "PIM_MAC_AB"
        else:  # bg
            K_steps = math.ceil(L / n_pch / n_rank / n_bg)
            per_k_col_factor = n_mac
            mac_opcode = "PIM_MAC_SB"

        process_k_indices(K_steps, per_k_col_factor, mac_opcode)

    def softmax(L: int):
        if page_scheduling and selected_pages is not None:
            # Page scheduling: process only the selected pages for this row
            # Generate SFM for the entire block (not per page)
            pages_processed = len(selected_pages)
            print(f"    Row-based SFM: Processing {pages_processed} pages for current row (time_step={time_step}, layer={layer})")
            
            # Generate SFM commands for the entire block (channel-wise, not page-wise)
            for lch in range(math.ceil(valid_channel)):
                # Use the first page's address as base (since SFM processes the whole channel)
                addr = lch * HBM_GS['ch'] 
                cmd_sfm[itr].append(f"PIM_SFM 0x{hex(addr)[2:]:0>8}")
        else:
            # Traditional scheduling: process all pages at once
            for lch in range(math.ceil(valid_channel)):
                addr = lch * HBM_GS['ch']
                cmd_sfm[itr].append(f"PIM_SFM 0x{hex(addr)[2:]:0>8}")

    # generate
    score_cpvec(key_addr, L)
    score_mac(key_addr, L)
    softmax(L)
    context_cpvec(val_addr, L)
    context_mac(val_addr, L)

# Global deduplication sets for page-first mode
global_wrgb_issued = set()
global_mvsb_issued = set()

def run_attention(dhead_val: int, n_head_per_hbm: int, L: int, trace_file_name: str,
                  page_size: int, selected_pages: Optional[List[int]],
                  csv_data: Optional[Dict] = None, page_scheduling: bool = False, 
                  csv_time_steps: Optional[int] = None, time_steps: Optional[List[int]] = None,
                  is_global_session: bool = False):
    """Run attention with traditional overlapping schedule"""
    global dhead, n_mac, global_wrgb_issued, global_mvsb_issued
    dhead = dhead_val
    n_mac = int(HBM_GS['col'] / data_size)

    # Reset global deduplication for new session
    if is_global_session:
        global_wrgb_issued = set()
        global_mvsb_issued = set()

    partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
    v_offset = 2 ** 23

    cmd_list_reset()

    num_layers = 32
    layer_offset = 2 ** 25

    # How many time steps to simulate: use CSV time steps if available, else max_L
    # Determine which time steps to process
    if time_steps is not None:
        # Use provided time_steps list (for page-first scheduling)
        process_time_steps = time_steps
        print(f"Processing {len(process_time_steps)} specified time steps: {process_time_steps} (mac_mode={mac_mode})")
    elif csv_data and csv_time_steps:
        process_time_steps = list(range(csv_time_steps))
        print(f"Processing {len(process_time_steps)} time steps from CSV data (mac_mode={mac_mode})")
    else:
        process_time_steps = list(range(max_L))
        print(f"Processing {len(process_time_steps)} time steps (default, mac_mode={mac_mode})")

    global_itr_counter = 0
    for time_step in process_time_steps:
        print(f"Time Step {time_step}: processing layers 0..{num_layers-1}")
        for layer in range(num_layers):
            print(f"  Processing Layer {layer}")
            layer_base_addr = layer * layer_offset + time_step * (num_layers * layer_offset)
            num_itr = math.ceil(n_head_per_hbm / n_channel)
            for itr in range(num_itr):
                remainder = 0
                if (n_head_per_hbm / ((itr + 1) * n_channel) < 1):
                    remainder = n_head_per_hbm % n_channel
                key_addr = layer_base_addr + global_itr_counter * partition_size
                val_addr = key_addr + v_offset
                head_base = itr * n_channel
                if remainder == 0:
                    Attention(L, key_addr, val_addr, global_itr_counter, page_size, selected_pages, csv_data,
                             n_channel, layer, time_step, page_scheduling, head_base=head_base, csv_time_steps=csv_time_steps)
                else:
                    Attention(L, key_addr, val_addr, global_itr_counter, page_size, selected_pages, csv_data,
                             remainder, layer, time_step, page_scheduling, head_base=head_base, csv_time_steps=csv_time_steps)
                global_itr_counter += 1

    # Overlap stitching (adaptive to actual filtered commands)
    barrier = []
    for lch in range(n_channel):
        addr = lch * HBM_GS['ch']
        barrier.append(f"PIM_BARRIER 0x{hex(addr)[2:]:0>8}")

    total_cmd = []
    num_itr = math.ceil(n_head_per_hbm / n_channel)
    for i in range(0, max(0, num_itr - 1), 2):
        # Head0: Score
        if i < len(cmd_score_wrgb):
            total_cmd += cmd_score_wrgb[i]
        if i == 0 and i < len(cmd_score_mac) and len(cmd_score_mac[i]) > 0 and len(cmd_score_mac[i][0]) > 0:
            for j in range(valid_channels[i]):
                if j < len(cmd_score_mac[i][0]):
                    total_cmd.append(cmd_score_mac[i][0][j])
        total_cmd += barrier

        # Dynamic length based on actual MAC commands generated
        actual_mac_length = len(cmd_score_mac[i]) if i < len(cmd_score_mac) else 0
        if actual_mac_length == 0:
            length = 0  # Skip empty loops to prevent barrier explosion
        else:
            length = max(1, math.ceil(actual_mac_length / 16))
        
        if length > 0:  # Only process if there are actual commands
            for j in range(0, length + 1):
                if i < len(cmd_score_mac) and not j == length:
                    stride = 16
                    for k in range(stride):
                        idx = j * stride + k
                        if idx >= len(cmd_score_mac[i]):
                            break
                        total_cmd += cmd_score_mac[i][idx]
                if i < len(cmd_score_mvsb) and not j == 0 and (j - 1) < len(cmd_score_mvsb[i]):
                    total_cmd += cmd_score_mvsb[i][j - 1]
                if (i + 1) < len(cmd_score_wrgb) and not j == length:
                    stride = int(math.ceil(dhead / n_mac) * math.ceil(valid_channels[i + 1]) / max(1, length))
                    for k in range(stride):
                        idx = j * stride + k
                        if idx >= len(cmd_score_wrgb[i + 1]):
                            break
                        total_cmd.append(cmd_score_wrgb[i + 1][idx])
                if not j == length:
                    total_cmd += barrier

        # Head0: SoftMax, Head1: Score
        # Use same dynamic length as calculated above
        actual_mac_length_next = len(cmd_score_mac[i + 1]) if (i + 1) < len(cmd_score_mac) else 0
        if actual_mac_length_next == 0:
            length_next = 0  # Skip empty loops
        else:
            length_next = max(1, math.ceil(actual_mac_length_next / 16))
        
        if length_next > 0:  # Only process if there are actual commands
            for j in range(0, length_next + 1):
                if (i + 1) < len(cmd_score_mac) and not j == length_next:
                    stride = 16
                    for k in range(stride):
                        idx = j * stride + k
                        if idx >= len(cmd_score_mac[i + 1]):
                            break
                        total_cmd += cmd_score_mac[i + 1][idx]
                if (i + 1) < len(cmd_score_mvsb) and not j == 0 and (j - 1) < len(cmd_score_mvsb[i + 1]):
                    total_cmd += cmd_score_mvsb[i + 1][j - 1]
                if i < len(cmd_sfm) and j == 0:
                    total_cmd += cmd_sfm[i]
                if i < len(cmd_context_mvgb) and not j == length_next:
                    if j >= math.floor(length_next / 2):
                        stride = int(math.ceil(L / (n_pch * n_mac)) * math.ceil(valid_channels[i]) / max(1, math.ceil(length_next / 2)))
                        for k in range(stride):
                            idx = (j - math.floor(length_next / 2)) * stride + k
                            if idx >= len(cmd_context_mvgb[i]):
                                break
                            total_cmd.append(cmd_context_mvgb[i][idx])
                if not j == length_next:
                    total_cmd += barrier

        # Head0: Context, Head1: Softmax
        # Dynamic length2 based on actual context MAC commands
        actual_context_length = len(cmd_context_mac[i]) if i < len(cmd_context_mac) else 0
        if actual_context_length == 0:
            length2 = 0  # Skip empty loops
        else:
            length2 = max(1, actual_context_length)
        
        if length2 > 0:  # Only process if there are actual commands
            for j in range(0, length2 + 1):
                if i < len(cmd_context_mac) and not j == length2 and j < len(cmd_context_mac[i]):
                    total_cmd += cmd_context_mac[i][j]
                if i < len(cmd_context_mvsb) and not j == 0 and (j - 1) < len(cmd_context_mvsb[i]):
                    total_cmd += cmd_context_mvsb[i][j - 1]
                if (i + 1) < len(cmd_sfm) and j == 0:
                    total_cmd += cmd_sfm[i + 1]
                if (i + 1) < len(cmd_context_mvgb) and not j == length2:
                    if j >= math.floor(length2 / 2):
                        stride = int(math.ceil(L / (n_pch * n_mac)) * math.ceil(valid_channels[i + 1]) / max(1, math.ceil(length2 / 2)))
                        for k in range(stride):
                            idx = (j - math.floor(length2 / 2)) * stride + k
                            if idx >= len(cmd_context_mvgb[i + 1]):
                                break
                            total_cmd.append(cmd_context_mvgb[i + 1][idx])
                if not j == length2:
                    total_cmd += barrier

        # Head1: Context
        if length2 > 0:  # Only process if there are actual commands
            for j in range(0, length2 + 1):
                if (i + 1) < len(cmd_context_mac) and not j == length2 and j < len(cmd_context_mac[i + 1]):
                    total_cmd += cmd_context_mac[i + 1][j]
                if (i + 1) < len(cmd_context_mvsb) and not j == 0 and (j - 1) < len(cmd_context_mvsb[i + 1]):
                    total_cmd += cmd_context_mvsb[i + 1][j - 1]
                if not j == length2:
                    total_cmd += barrier

    if num_itr % 2 != 0:
        i = num_itr - 1
        if i < len(cmd_score_wrgb):
            total_cmd += cmd_score_wrgb[i]
        total_cmd += barrier

        # Dynamic length for odd iteration
        actual_mac_length_odd = len(cmd_score_mac[i]) if i < len(cmd_score_mac) else 0
        if actual_mac_length_odd == 0:
            length = 0  # Skip empty loops
        else:
            length = max(1, math.ceil(actual_mac_length_odd / 16))
        
        if length > 0:  # Only process if there are actual commands
            for j in range(0, length + 1):
                if i < len(cmd_score_mac) and not j == length:
                    stride = 16
                    for k in range(stride):
                        idx = j * stride + k
                        if idx >= len(cmd_score_mac[i]):
                            break
                        total_cmd += cmd_score_mac[i][idx]
                if i < len(cmd_score_mvsb) and not j == 0 and (j - 1) < len(cmd_score_mvsb[i]):
                    total_cmd += cmd_score_mvsb[i][j - 1]
                if not j == length:
                    total_cmd += barrier

        if i < len(cmd_sfm):
            total_cmd += cmd_sfm[i]
        if i < len(cmd_context_mvgb):
            total_cmd += cmd_context_mvgb[i]
        total_cmd += barrier

        # Context for odd iteration
        actual_context_length_odd = len(cmd_context_mac[i]) if i < len(cmd_context_mac) else 0
        if actual_context_length_odd == 0:
            length2 = 0  # Skip empty loops
        else:
            length2 = max(1, actual_context_length_odd)
        
        if length2 > 0:  # Only process if there are actual commands
            for j in range(0, length2 + 1):
                if i < len(cmd_context_mac) and not j == length2 and j < len(cmd_context_mac[i]):
                    total_cmd += cmd_context_mac[i][j]
                if i < len(cmd_context_mvsb) and not j == 0 and (j - 1) < len(cmd_context_mvsb[i]):
                    total_cmd += cmd_context_mvsb[i][j - 1]
                if not j == length2:
                    total_cmd += barrier

    with open(trace_file_name, "w") as f:
        for cmd in total_cmd:
            f.write(cmd + "\n")
    return total_cmd

# =========================
# Page-first scheduler
# =========================

def build_page_blocks(L: int, page_size: int, block_pages: int = 8):
    n = num_pages(L, page_size)
    blocks = []
    for start in range(0, n, block_pages):
        blk = list(range(start, min(start + block_pages, n)))
        blocks.append(blk)
    return blocks

def run_attention_pagefirst_clean(
    dhead_val: int, n_head_per_hbm: int, L: int, trace_file_name: str,
    page_size: int, selected_pages: Optional[List[int]] = None,
    csv_data: Optional[Dict] = None, block_pages: int = 8, csv_time_steps: Optional[int] = None
):
    print("🚀 Row-based page scheduling (QK→SFM→V per page row)")
    
    # Calculate pages per row (K-row 기준)
    pages_per_row = 4  # K region에서 1 row = 4 pages
    
    if selected_pages is not None:
        s = sorted(selected_pages)
        # Group pages into K-rows of 4 pages each for better deduplication
        rows = [s[i:i+pages_per_row] for i in range(0, len(s), pages_per_row)]
        print(f"   Using {len(rows)} K-rows from selected pages ({len(s)} pages)")
        print(f"   Pages per K-row: {pages_per_row}")
    else:
        # Create K-rows from all pages
        total_pages = math.ceil(L / page_size)
        all_pages = list(range(total_pages))
        rows = [all_pages[i:i+pages_per_row] for i in range(0, len(all_pages), pages_per_row)]
        print(f"   Using {len(rows)} K-rows from all pages ({total_pages} pages)")
        print(f"   Pages per K-row: {pages_per_row}")

    total_cmd_all = []
    
    # Reset global deduplication at the start of page-first processing
    global global_wrgb_issued, global_mvsb_issued
    global_wrgb_issued = set()
    global_mvsb_issued = set()
    
    # Process each K-row as a single unit
    # K-row 기준 처리: QK for all pages in K-row → SFM for K-row → V for all pages in K-row
    for row_idx, page_row in enumerate(rows):
        print(f"   Processing K-row {row_idx}: pages {page_row[0]}~{page_row[-1]} ({len(page_row)} pages)")
        
        temp_trace = f"temp_trace_krow_{row_idx}.trace"
        # Process the entire K-row at once with global session flag
        run_attention(dhead_val, n_head_per_hbm, L, temp_trace, page_size, page_row, csv_data, 
                     page_scheduling=True, csv_time_steps=csv_time_steps, is_global_session=False)
        
        if os.path.exists(temp_trace):
            with open(temp_trace, "r") as f:
                cmds = [line.strip() for line in f if line.strip().startswith("PIM_")]
            total_cmd_all.extend(cmds)
            os.remove(temp_trace)
            print(f"   K-row {row_idx}: {len(cmds)} commands generated")
        else:
            print(f"   ⚠️ K-row {row_idx}: no commands produced")

    with open(trace_file_name, "w") as f:
        for c in total_cmd_all:
            f.write(c + "\n")
    print(f"✅ K-row based page scheduling completed: {trace_file_name}")
    print(f"📊 Global deduplication stats:")
    print(f"   WR_GB addresses deduplicated: {len(global_wrgb_issued)}")
    print(f"   MV_SB addresses deduplicated: {len(global_mvsb_issued)}")
    return total_cmd_all

# =========================
# CLI
# =========================

def main():
    global dhead, max_L, data_size, n_mac, mac_mode, mod_alignment_score, mod_alignment_context

    p = argparse.ArgumentParser(description="Page-aware AttAcc trace generator with CSV support",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-dh", "--dhead", type=int, default=128)
    p.add_argument("-nh", "--nhead", type=int, default=32)
    p.add_argument("-l",  "--seqlen", type=int, default=21504)
    p.add_argument("-maxl","--maxlen", type=int, default=4)
    p.add_argument("-db", "--dbyte", type=int, default=2)
    p.add_argument("-o",  "--output", type=str, default="mos_buffer_page.trace")

    # page selection
    p.add_argument("-ps","--page_size", type=int, default=64)
    p.add_argument("-csv","--csv_file", type=str, default="")
    p.add_argument("-pm","--page_mode", type=str, default="all",
                   choices=['all','even','odd','first_half','second_half','exclude_back_half'])
    p.add_argument("-pl","--page_list", type=str, default="")
    p.add_argument("--page_scheduling", type=str, default="false",
                   choices=['true','false','True','False'])
    p.add_argument("--mac_mode", type=str, default="buffer", choices=['buffer','bank','bg'])

    # MOD alignment
    p.add_argument("--mod_score", type=int, default=1, choices=[1,2,8])
    p.add_argument("--mod_context", type=int, default=1, choices=[1,2,8])

    args = p.parse_args()

    dhead = args.dhead
    max_L = args.maxlen
    L = args.seqlen
    n_head_per_hbm = args.nhead
    page_size = args.page_size
    data_size = args.dbyte
    n_mac = int(HBM_GS['col'] / data_size)
    mac_mode = args.mac_mode
    mod_alignment_score   = args.mod_score
    mod_alignment_context = args.mod_context
    page_scheduling = args.page_scheduling.lower() in ['true','1','yes','on']

    print("------   Make a trace of page-aware AttAcc   ------")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("---------------------------------------------------")

    # CSV & page selection
    selected_pages = None
    csv_data = None
    csv_time_steps = None
    if args.csv_file:
        print(f"Loading page selection from CSV: {args.csv_file}")
        csv_data, csv_time_steps = load_page_selection_data(args.csv_file)
        if csv_data:
            print(f"Loaded CSV data entries: {len(csv_data)}")
            print(f"CSV time_steps: {csv_time_steps}")
            all_pages = set()
            for pages in csv_data.values():
                all_pages.update(pages)
            print(f"CSV contains {len(all_pages)} unique pages total")
        else:
            print("No CSV data loaded, will use static page_mode")

    if csv_data is None:
        if args.page_list:
            try:
                page_indices = [int(x.strip()) for x in args.page_list.split(',')]
                selected_pages = build_selected_pages('only_list', L, page_size, include_pages=page_indices)
            except ValueError:
                print("Error parsing page_list; fallback to 'all'")
                selected_pages = None
        else:
            selected_pages = build_selected_pages(args.page_mode, L, page_size)
        print(f"Static page selection: {selected_pages if selected_pages is not None else 'ALL'}")
        if selected_pages:
            print(f"Selected {len(selected_pages)} / {num_pages(L, page_size)} pages")
    else:
        # CSV data is loaded - extract unique pages from CSV
        unique_pages = set()
        for key in csv_data:
            layer, head, time_step = key
            pages = csv_data[key]  # pages are stored directly as values
            if pages:
                unique_pages.update(pages)
        
        if unique_pages:
            selected_pages = sorted(list(unique_pages))
            print(f"CSV-based page selection: {len(selected_pages)} unique pages from CSV data")
            print(f"Page range: {min(selected_pages)}~{max(selected_pages)}")
        else:
            selected_pages = None
            print("No pages found in CSV data, using all pages")

    # Run
    if page_scheduling:
        print("🚀 TRUE page-first scheduling ENABLED")
        # Initialize global deduplication before starting
        global global_wrgb_issued, global_mvsb_issued
        global_wrgb_issued = set()
        global_mvsb_issued = set()
        run_attention_pagefirst_clean(dhead, n_head_per_hbm, L, args.output, page_size, selected_pages, csv_data, csv_time_steps=csv_time_steps)
    else:
        print("📊 Traditional scheduling (all-K → SFM → all-V)")
        run_attention(dhead, n_head_per_hbm, L, args.output, page_size, selected_pages, csv_data, page_scheduling=False, csv_time_steps=csv_time_steps, is_global_session=True)

    print(f"Trace generation completed: {args.output}")

    # Save analyses
    print("\n=== Mapping Analysis ===")
    analyzer.current_layer = 0
    analyzer.current_time_step = 0
    analyzer.save_logs()

    print("\n=== Row Activation Analysis ===")
    row_analyzer.save_activation_analysis()

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()

