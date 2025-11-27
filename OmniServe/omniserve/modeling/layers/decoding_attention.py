from typing import Dict, List, Optional

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from datetime import datetime
from torch.nn.functional import cosine_similarity

# NOTE:
# EXPERIMENT_MODE controls the page-selection strategy:
#   - Mode 0: Select pages matching a specific modulo condition (multiples of MULTIPLE_VALUE)
#   - Mode 1: Use rule-based selection (predefined mathematical patterns)
#   - Other modes: Reserved for additional strategies

# Rule-based modes (EXPERIMENT_MODE = 1) support:
#   WnRm patterns: W4R8, W2R4, W2R8, W1R2, W1R4, W1R8
#     - n = Write modulus (Wn or WM_n)
#     - m = Read modulus  (Rm or RM_m)

# Environment variables provide default hyperparameters.
# These are usually set in the shell script (niah_test.sh).
# If not set, fallback values are applied here.
EXPERIMENT_MODE = int(os.environ.get("EXPERIMENT_MODE", 0))
MULTIPLE_VALUE = int(os.environ.get("MULTIPLE_VALUE", 4))  # Fallback: 4 (should be set by shell script)
MODULO_VALUE = int(os.environ.get("MODULO_VALUE", 0))      # Default: 0 select pages where page_idx % MULTIPLE_VALUE == 0
FILL_REMAINING = os.environ.get("FILL_REMAINING", "true").lower() == "true"  # Fallback: true (x% should be set by shell script)
MULTIPLE_PERCENTAGE = int(os.environ.get("MULTIPLE_PERCENTAGE", 90))  # Fallback: 90% (alpha should be set by shell script)
RULE_BASED_MODE = os.environ.get("RULE_BASED_MODE", "W4R8")  # Default: W4R8


import omniserve_backend.fused_attention_fine_grained_dense as fused_attention_fine_grained_dense
import omniserve_backend.fused_attention_fine_grained_sparse as fused_attention_fine_grained_sparse
import omniserve_backend.fused_attention_per_tensor_dense as fused_attention_per_tensor_dense
import omniserve_backend.fused_attention_per_tensor_sparse as fused_attention_per_tensor_sparse
import omniserve_backend.fused_attention_selector as fused_attention_selector
import omniserve_backend.fused_attention_pure_dense as fused_attention_pure_dense


class DecodingAttentionWrapper(torch.nn.Module):
    # Class-level variables for unified logging
    _unified_log_file = None
    _unified_log_lock = None
    
    def __init__(
        self,
        layer_idx: int,
        static_sparsity_enabled: bool,
        head_dim: int,
        alibi_slopes: int,
        memory_max_len: int,
        tokens_per_block: int,
        rotary_embedding_dim: int,
        rotary_base: int, 
        rope_scaling: Dict,
        neox_rotary_style: bool,
        kv_quant_granularity: str,
        kv_cache_config: Dict,
        use_int8: bool,
        sparse_decode_mode: int,
        sub_chunk_size: int,
        dynamic_sparse_token_budget: int,
        multiblock_switch: int,
        selector_update_interval: int,
        ):
        super().__init__()

        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.alibi_slopes = alibi_slopes
        self.memory_max_len = memory_max_len
        self.tokens_per_block = tokens_per_block
        self.rotary_embedding_dim = rotary_embedding_dim
        self.rotary_base = rotary_base
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None:
            self.rope_scaling_factor = self.rope_scaling["factor"]
            assert self.rope_scaling["type"] == "linear", f"Unsupported rope scaling type {self.rope_scaling['type']}"
        else:
            self.rope_scaling_factor = 1.0
        self.neox_rotary_style = neox_rotary_style
        self.kv_quant_granularity = kv_quant_granularity
        self.kv_cache_config = kv_cache_config
        self.use_int8 = use_int8

        self.sparse_decode_mode = sparse_decode_mode
        self.sub_chunk_size = sub_chunk_size
        self.dynamic_sparse_token_budget = dynamic_sparse_token_budget
        self.multiblock_switch = multiblock_switch
        self.selector_update_interval = selector_update_interval
        
        # Token coherence analysis state
        self.coherence_window_sizes = [16, 32, 64, 128]
        self.coherence_results = {size: [] for size in self.coherence_window_sizes}
        self.analysis_count = 0
        
        # CSV logging for selected pages (controlled by environment variable)
        enable_log_env = os.environ.get("ENABLE_PAGE_SELECTION_LOG", "false")
        self.enable_csv_logging = enable_log_env.lower() == "true"
        
        # Score logging (controlled by environment variable)
        enable_score_log_env = os.environ.get("ENABLE_SCORE_LOG", "false")
        self.enable_score_logging = enable_score_log_env.lower() == "true"
        
        print(f"[DEBUG] Layer {layer_idx}: ENABLE_PAGE_SELECTION_LOG = '{enable_log_env}', enable_csv_logging = {self.enable_csv_logging}")
        print(f"[DEBUG] Layer {layer_idx}: ENABLE_SCORE_LOG = '{enable_score_log_env}', enable_score_logging = {self.enable_score_logging}")
        
        # Setup log directory and timestamp
        enable_unified_log_env = os.environ.get("ENABLE_UNIFIED_LOG", "false")
        enable_unified_log = enable_unified_log_env.lower() == "true"
        
        if self.enable_csv_logging or self.enable_score_logging or enable_unified_log:
            self.csv_log_dir = os.environ.get("PAGE_SELECTION_LOG_DIR", "page_selection_logs")
            os.makedirs(self.csv_log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if self.enable_csv_logging:
                self.csv_filename = f"{self.csv_log_dir}/layer_{layer_idx}_exp{EXPERIMENT_MODE}_{timestamp}.csv"
                print(f"[DEBUG] Layer {layer_idx}: Page selection logging enabled, file: {self.csv_filename}")
            
            if self.enable_score_logging:
                self.score_csv_filename = f"{self.csv_log_dir}/layer_{layer_idx}_scores_exp{EXPERIMENT_MODE}_{timestamp}.csv"
                print(f"[DEBUG] Layer {layer_idx}: Score logging enabled, file: {self.score_csv_filename}")
                
            # Initialize unified log file (shared across all layers)
            if enable_unified_log and DecodingAttentionWrapper._unified_log_file is None:
                import threading
                DecodingAttentionWrapper._unified_log_lock = threading.Lock()
                unified_filename = f"{self.csv_log_dir}/page_selection_log_{timestamp}_selected_pages.csv"
                DecodingAttentionWrapper._unified_log_file = unified_filename
                print(f"[DEBUG] Unified logging enabled, file: {unified_filename}")
        else:
            print(f"[DEBUG] Layer {layer_idx}: All logging disabled")
        
        if self.sparse_decode_mode != 0:
            if kv_quant_granularity == "per_tensor":
                self.forward = self.forward_w_dynamic_sparse_per_tensor
            elif kv_quant_granularity == "fine_grained":
                self.forward = self.forward_w_dynamic_sparse_fine_grained
            else:
                raise NotImplementedError(f"Unsupported kv_quant_granularity {kv_quant_granularity}")
        else:
            if static_sparsity_enabled:
                if kv_quant_granularity == "per_tensor":
                    self.forward = self.forward_wo_dynamic_sparse_per_tensor
                elif kv_quant_granularity == "fine_grained":
                    self.forward = self.forward_wo_dynamic_sparse_fine_grained
                else:
                    raise NotImplementedError(f"Unsupported kv_quant_granularity {kv_quant_granularity}")
            else:
                if kv_quant_granularity == "fine_grained":
                    self.forward = self.forward_pure_dense
                elif kv_quant_granularity == "per_tensor":
                    # raise NotImplementedError("per_tensor kv_quant_granularity is not supported for pure dense attention")
                    self.forward = self.forward_wo_dynamic_sparse_per_tensor    # NOTE: Per_tensor pure dense is has not been implemented yet. Just use the forward_wo_dynamic_sparse_per_tensor sparse for now.
                else:
                    raise NotImplementedError(f"Unsupported kv_quant_granularity {kv_quant_granularity}")
                


    @torch.no_grad()
    def analyze_token_coherence(self, k, v, timestep):
        """
        Analyze token coherence within different window sizes.
        Measures how similar tokens are within sliding windows of different sizes.
        """
        # Token coherence analysis disabled - return early
        return
            
        try:
            # Extract token embeddings from key vectors (using first head for simplicity)
            if k.dim() == 4:  # [batch, heads, seq_len, head_dim]
                tokens = k[0, 0, :timestep, :]  # [seq_len, head_dim]
            elif k.dim() == 3:  # [batch, seq_len, head_dim]
                tokens = k[0, :timestep, :]
            else:
                return
                
            seq_len = tokens.shape[0]
            
            # Normalize tokens for better cosine similarity calculation
            tokens = torch.nn.functional.normalize(tokens, dim=1)
            
            # Analyze coherence for each window size
            for window_size in self.coherence_window_sizes:
                if seq_len < window_size:
                    continue
                    
                coherence_scores = []
                
                # Use fewer windows for efficiency (every window_size/2 tokens)
                step_size = max(1, window_size // 2)
                for start_idx in range(0, seq_len - window_size + 1, step_size):
                    end_idx = start_idx + window_size
                    window_tokens = tokens[start_idx:end_idx]  # [window_size, head_dim]
                    
                    # Calculate average pairwise cosine similarity more efficiently
                    # Using matrix operations instead of nested loops
                    similarity_matrix = torch.mm(window_tokens, window_tokens.t())  # [window_size, window_size]
                    
                    # Extract upper triangular part (excluding diagonal)
                    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
                    similarities = similarity_matrix[mask]
                    
                    if similarities.numel() > 0:
                        avg_similarity = similarities.mean().item()
                        coherence_scores.append(avg_similarity)
                
                # Store the average coherence for this window size
                if coherence_scores:
                    avg_coherence = sum(coherence_scores) / len(coherence_scores)
                    self.coherence_results[window_size].append(avg_coherence)
                    
                    # Print immediate results for debugging
                    if len(self.coherence_results[window_size]) == 1:  # First measurement
                        print(f"[Coherence] Window {window_size}: {avg_coherence:.4f}")
            
            self.analysis_count += 1
            
            # Generate plot every 50 analyses (more frequent for better monitoring)
            if self.analysis_count % 50 == 0:
                self.plot_coherence_results()
                
        except Exception as e:
            print(f"Error in token coherence analysis: {e}")
    
    def plot_coherence_results(self):
        """Plot coherence analysis results"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(12, 8))
            
            # Prepare data for plotting
            window_sizes_plot = []
            mean_coherences = []
            std_coherences = []
            
            print(f"\n=== Token Coherence Analysis Results (Step {self.analysis_count}) ===")
            
            for window_size in self.coherence_window_sizes:
                if self.coherence_results[window_size]:
                    # Calculate statistics
                    scores = self.coherence_results[window_size]
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    window_sizes_plot.append(window_size)
                    mean_coherences.append(mean_score)
                    std_coherences.append(std_score)
                    
                    print(f"Window size {window_size:3d}: Mean coherence = {mean_score:.4f} ± {std_score:.4f} (n={len(scores)})")
            
            if len(window_sizes_plot) >= 2:
                # Main plot: Mean coherence vs window size
                plt.subplot(2, 1, 1)
                plt.errorbar(window_sizes_plot, mean_coherences, yerr=std_coherences, 
                           marker='o', markersize=8, capsize=5, linewidth=2, capthick=2)
                plt.xlabel('Window Size (tokens)')
                plt.ylabel('Token Coherence (Cosine Similarity)')
                plt.title(f'Token Coherence vs Window Size (Analysis Step {self.analysis_count})')
                plt.grid(True, alpha=0.3)
                plt.xticks(window_sizes_plot)
                
                # Trend analysis
                if mean_coherences[-1] < mean_coherences[0]:
                    trend_text = "↓ Decreasing (Expected)"
                    plt.text(0.02, 0.98, trend_text, transform=plt.gca().transAxes, 
                            verticalalignment='top', fontsize=10, color='green')
                else:
                    trend_text = "↑ Increasing (Unexpected)"
                    plt.text(0.02, 0.98, trend_text, transform=plt.gca().transAxes, 
                            verticalalignment='top', fontsize=10, color='red')
                
                # Distribution plot
                plt.subplot(2, 1, 2)
                box_data = [self.coherence_results[size] for size in window_sizes_plot]
                plt.boxplot(box_data, positions=range(len(window_sizes_plot)), tick_labels=window_sizes_plot)
                plt.xlabel('Window Size (tokens)')
                plt.ylabel('Coherence Distribution')
                plt.title('Distribution of Coherence Scores')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                filename = f'token_coherence_layer{self.layer_idx}_step{self.analysis_count}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Coherence analysis plot saved as {filename}")
                
                # Analysis summary
                coherence_64 = mean_coherences[2] if len(mean_coherences) > 2 else None
                optimal_idx = np.argmax(mean_coherences)
                optimal_size = window_sizes_plot[optimal_idx]
                
                print(f"Current page size (64 tokens) coherence: {coherence_64:.4f}" if coherence_64 else "64-token data not available")
                print(f"Optimal window size: {optimal_size} tokens (coherence: {mean_coherences[optimal_idx]:.4f})")
                print("=" * 60)
            
        except Exception as e:
            print(f"Error plotting coherence results: {e}")
            # Fallback: just print the numerical results
            print(f"\n=== Token Coherence Analysis Results (Step {self.analysis_count}) ===")
            for window_size in self.coherence_window_sizes:
                if self.coherence_results[window_size]:
                    scores = self.coherence_results[window_size]
                    print(f"Window size {window_size:3d}: Mean coherence = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    @torch.no_grad()
    def dynamic_select_topk_pages(
        self, 
        q, k, v,
        retrieval_block_tables, streaming_block_tables, retrieval_head_flags, head_rank_table,
        lengths_per_sample, sink_size, local_size, sink_blocks, local_blocks,
        size_per_retrieval_token, size_per_streaming_token,
        num_retrieval_kv_heads, num_streaming_kv_heads, timestep, hidden_dim_per_retrieval_token
    ):
        # Token coherence analysis disabled - EXPERIMENT_MODE 4 now used for multiple selection
        # if EXPERIMENT_MODE == 4:
        #     self.analyze_token_coherence(k, v, timestep)
        
        if timestep <= self.dynamic_sparse_token_budget:
            selected_page_idx = torch.range(0, timestep // self.tokens_per_block, device=q.device, dtype=torch.int32).unsqueeze(0).unsqueeze(0).expand(q.shape[0], q.shape[1], -1).contiguous()
        
        else:
            dynamic_sparse_token_budget = min(self.dynamic_sparse_token_budget, timestep)
            selected_page_stats = fused_attention_selector.single_query_page_selector(
                q,
                k,
                v,                               # Actually of no use. (just keep for the interface)
                retrieval_block_tables,
                streaming_block_tables,          # Actually of no use. (just keep for the interface)
                retrieval_head_flags,
                head_rank_table,
                None,                            # selected_page_idx: This is of no use (just keep for the interface)
                lengths_per_sample,
                self.alibi_slopes,
                self.memory_max_len,
                self.tokens_per_block,
                size_per_retrieval_token,
                size_per_streaming_token,        # Actually of no use. (just keep for the interface)
                sink_size, local_size,           # Actually of no use. (just keep for the interface)
                sink_blocks, local_blocks,       # Actually of no use. (just keep for the interface)
                num_retrieval_kv_heads, 
                num_streaming_kv_heads,          # Actually of no use. (just keep for the interface)
                timestep,                        # NOTE (shang): timestep is the length of history, not including the current token! 
                self.rotary_embedding_dim,
                self.rotary_base,
                self.rope_scaling_factor,
                self.neox_rotary_style,
                self.kv_cache_config["INT4_ENABLED"],
                True, # self.kv_cache_config["ZEROS_ENABLED"],     # TODO: Fix this error for buffer offset.
                self.sub_chunk_size,
                hidden_dim_per_retrieval_token,
                1000000,                         # const int multiblock_switch  # FIXME: Currently never activate it in page selector!
            )
 
            selected_page_stats = selected_page_stats.view(q.shape[0], q.shape[1], -1, self.tokens_per_block // self.sub_chunk_size)
            selected_page_stats = torch.max(selected_page_stats, dim=-1).values        # max over sub-chunk-dim


            total_page_num = selected_page_stats.size(-1)
            
            # Adjust target_k for 32-token emulation in experiment mode 5
            if EXPERIMENT_MODE == 5 and EXP5_EMULATION_MODE == "32":
                # For 32-token emulation, we need 2x pages since we use only half of each page
                effective_budget = dynamic_sparse_token_budget * 2
                target_k = min(max(3, effective_budget // self.tokens_per_block), total_page_num) - 1
                print(f"[Experiment Mode 5] 32-token emulation: effective_budget={effective_budget}, target_k={target_k}")
            else:
                target_k = min(max(3, dynamic_sparse_token_budget // self.tokens_per_block), total_page_num) - 1
             
            if EXPERIMENT_MODE == 0:
                fill_mode = "fill remaining with high-score pages" if FILL_REMAINING else "only select multiple pages"
                # print(f"[Experiment Mode 0] Multiple selection priority - selecting multiples of {MULTIPLE_VALUE} with modulo {MODULO_VALUE}, {fill_mode}, total pages: {total_page_num}")

                page_stats = selected_page_stats[:, :, :-1]  # [batch, heads, pages-1]
                page_scores, page_indices = page_stats.topk(k=page_stats.size(-1), dim=-1)
                
                # Store scores for logging
                self._current_page_scores = page_scores
                self._current_page_indices = page_indices
                
                batch_size, num_heads, _ = page_indices.shape
                selected_page_idx = torch.zeros(batch_size, num_heads, target_k, device=page_indices.device, dtype=torch.long)
                
                for b in range(batch_size):
                    for h in range(num_heads):
                        indices = page_indices[b, h, :]
                        scores = page_scores[b, h, :]
                        
                        multiple_mask = indices % MULTIPLE_VALUE == MODULO_VALUE
                        non_multiple_mask = ~multiple_mask
                        
                        multiple_indices = indices[multiple_mask]
                        multiple_scores = scores[multiple_mask]
                        non_multiple_indices = indices[non_multiple_mask]
                        non_multiple_scores = scores[non_multiple_mask]
                        
                        selected_list = []
                        
                        target_multiple_slots = int(target_k * MULTIPLE_PERCENTAGE / 100.0)
                        target_highscore_slots = target_k - target_multiple_slots
                        
                        if len(multiple_indices) >= target_multiple_slots:
                            selected_list.extend(multiple_indices[:target_multiple_slots].tolist())
                            actual_multiple = target_multiple_slots
                            padding_needed = 0
                        else:
                            selected_list.extend(multiple_indices.tolist())
                            actual_multiple = len(multiple_indices)
                            padding_needed = target_multiple_slots - len(multiple_indices)
                            
                            if padding_needed > 0 and len(multiple_indices) > 0:
                                padding_value = multiple_indices[0].item() 
                                for _ in range(padding_needed):
                                    selected_list.append(padding_value)
                        
                        # if padding_needed > 0:
                        #     print(f"[Mode 4] Multiple pages (idx % {MULTIPLE_VALUE} == {MODULO_VALUE}): selected {actual_multiple}/{len(multiple_indices)} pages + {padding_needed} padding (target: {target_multiple_slots} slots)")
                        # else:
                        #     print(f"[Mode 4] Multiple pages (idx % {MULTIPLE_VALUE} == {MODULO_VALUE}): selected {actual_multiple}/{len(multiple_indices)} pages (target: {target_multiple_slots} slots)")
                        
                        # high-score pages
                        remaining = target_highscore_slots
                        if remaining > 0:
                            remaining_indices = []
                            remaining_scores = []
                            
                            if actual_multiple < len(multiple_indices):
                                remaining_indices.extend(multiple_indices[actual_multiple:].tolist())
                                remaining_scores.extend(multiple_scores[actual_multiple:].tolist())
                            
                            remaining_indices.extend(non_multiple_indices.tolist())
                            remaining_scores.extend(non_multiple_scores.tolist())
                            
                            if remaining_indices:
                                remaining_pairs = list(zip(remaining_indices, remaining_scores))
                                remaining_pairs.sort(key=lambda x: x[1], reverse=True)
                                
                                high_score_indices = [pair[0] for pair in remaining_pairs[:remaining]]
                                selected_list.extend(high_score_indices)
                        
                        if len(selected_list) >= target_k:
                            selected_page_idx[b, h, :] = torch.tensor(selected_list[:target_k], device=page_indices.device)
                        elif len(selected_list) > 0:
                            selected_page_idx[b, h, :len(selected_list)] = torch.tensor(selected_list, device=page_indices.device)
                            if len(selected_list) < target_k:
                                padding_value = selected_list[0]
                                for i in range(len(selected_list), target_k):
                                    selected_page_idx[b, h, i] = padding_value
                        else:
                            # for padding when no pages are selected
                            selected_page_idx[b, h, :] = 0
                
                selected_without_last = selected_page_idx[0, 0, :]
                multiple_count = (selected_without_last % MULTIPLE_VALUE == MODULO_VALUE).sum().item()
                non_multiple_count = target_k - multiple_count
                
                # if MULTIPLE_PERCENTAGE == 100:
                #     print(f"[Experiment Mode 4] Selected {multiple_count} multiple pages (% {MULTIPLE_VALUE} == {MODULO_VALUE}), {non_multiple_count} padding slots")
                # else:
                #     print(f"[Experiment Mode 4] Selected {multiple_count} multiple pages (% {MULTIPLE_VALUE} == {MODULO_VALUE}), {non_multiple_count} high-score pages")
                    
            elif EXPERIMENT_MODE == 1:
                def calculate_rule_index(n, rule_mode):
                    """Calculate index based on rule mode and position n"""
                    if rule_mode == "W4R8":
                        return 16 * math.floor((n - 1) / 2) + 4 * ((n - 1) % 2)
                    elif rule_mode == "W2R4":
                        return 8 * math.floor((n - 1) / 2) + 2 * ((n - 1) % 2)
                    elif rule_mode == "W2R8":
                        return 16 * math.floor((n - 1) / 2) + 2 * ((n - 1) % 2)
                    elif rule_mode == "W1R2":
                        return 4 * math.floor((n - 1) / 2) + 1 * ((n - 1) % 2)
                    elif rule_mode == "W1R4":
                        return 8 * math.floor((n - 1) / 2) + 1 * ((n - 1) % 2)
                    elif rule_mode == "W1R8":
                        return 16 * math.floor((n - 1) / 2) + 1 * ((n - 1) % 2)
                    else:
                        # Default to W4R8 if unknown rule
                        return 16 * math.floor((n - 1) / 2) + 4 * ((n - 1) % 2)
                
                fill_mode = "fill remaining with high-score pages" if FILL_REMAINING else "only select rule-based pages"
                print(f"[Experiment Mode 41] Rule-based set selection - rule: {RULE_BASED_MODE}, {fill_mode}, total pages: {total_page_num}")
                
                # 전체 페이지 stats와 indices 가져오기 (마지막 페이지 제외)
                page_stats = selected_page_stats[:, :, :-1]  # [batch, heads, pages-1]
                page_scores, page_indices = page_stats.topk(k=page_stats.size(-1), dim=-1)  # 모든 페이지 정렬
                
                # Store scores for logging
                self._current_page_scores = page_scores
                self._current_page_indices = page_indices
                
                batch_size, num_heads, _ = page_indices.shape
                selected_page_idx = torch.zeros(batch_size, num_heads, target_k, device=page_indices.device, dtype=torch.long)
                
                for b in range(batch_size):
                    for h in range(num_heads):
                        indices = page_indices[b, h, :]
                        scores = page_scores[b, h, :]
                        
                        rule_based_indices = set()
                        for n in range(1, total_page_num + 1):
                            rule_index = calculate_rule_index(n, RULE_BASED_MODE)
                            if rule_index < total_page_num:
                                rule_based_indices.add(rule_index)
                        
                        rule_mask = torch.tensor([idx.item() in rule_based_indices for idx in indices], device=indices.device)
                        non_rule_mask = ~rule_mask
                        
                        rule_indices = indices[rule_mask]
                        rule_scores = scores[rule_mask]
                        non_rule_indices = indices[non_rule_mask]
                        non_rule_scores = scores[non_rule_mask]
                        
                        selected_list = []
                        
                        target_rule_slots = int(target_k * MULTIPLE_PERCENTAGE / 100.0)
                        target_highscore_slots = target_k - target_rule_slots
                        
                        if len(rule_indices) >= target_rule_slots:
                            selected_list.extend(rule_indices[:target_rule_slots].tolist())
                            actual_rule = target_rule_slots
                            padding_needed = 0
                        else:
                            selected_list.extend(rule_indices.tolist())
                            actual_rule = len(rule_indices)
                            padding_needed = target_rule_slots - len(rule_indices)
                            
                            if padding_needed > 0 and len(rule_indices) > 0:
                                padding_value = rule_indices[0].item()
                                for _ in range(padding_needed):
                                    selected_list.append(padding_value)
                        
                        if padding_needed > 0:
                            print(f"[Mode 41] Rule-based pages ({RULE_BASED_MODE}): selected {actual_rule}/{len(rule_indices)} pages + {padding_needed} padding (target: {target_rule_slots} slots)")
                        else:
                            print(f"[Mode 41] Rule-based pages ({RULE_BASED_MODE}): selected {actual_rule}/{len(rule_indices)} pages (target: {target_rule_slots} slots)")
                        
                        # for high-score pages
                        remaining = target_highscore_slots
                        if remaining > 0:
                            remaining_indices = []
                            remaining_scores = []
                            
                            if actual_rule < len(rule_indices):
                                remaining_indices.extend(rule_indices[actual_rule:].tolist())
                                remaining_scores.extend(rule_scores[actual_rule:].tolist())
                            
                            remaining_indices.extend(non_rule_indices.tolist())
                            remaining_scores.extend(non_rule_scores.tolist())
                            
                            if remaining_indices:
                                remaining_pairs = list(zip(remaining_indices, remaining_scores))
                                remaining_pairs.sort(key=lambda x: x[1], reverse=True)
                                
                                high_score_indices = [pair[0] for pair in remaining_pairs[:remaining]]
                                selected_list.extend(high_score_indices)
                        
                        if len(selected_list) >= target_k:
                            selected_page_idx[b, h, :] = torch.tensor(selected_list[:target_k], device=page_indices.device)
                        elif len(selected_list) > 0:
                            selected_page_idx[b, h, :len(selected_list)] = torch.tensor(selected_list, device=page_indices.device)
                            if len(selected_list) < target_k:
                                padding_value = selected_list[0]
                                for i in range(len(selected_list), target_k):
                                    selected_page_idx[b, h, i] = padding_value
                        else:
                            # for padding when no pages are selected
                            selected_page_idx[b, h, :] = 0
                
                selected_without_last = selected_page_idx[0, 0, :]
                
                rule_based_count = 0
                rule_based_indices_set = set()
                for n in range(1, total_page_num + 1):
                    rule_index = calculate_rule_index(n, RULE_BASED_MODE)
                    if rule_index < total_page_num:
                        rule_based_indices_set.add(rule_index)
                
                for page_idx in selected_without_last:
                    if page_idx.item() in rule_based_indices_set:
                        rule_based_count += 1
                
                non_rule_count = target_k - rule_based_count
                
                if MULTIPLE_PERCENTAGE == 100:
                    print(f"[Experiment Mode 41] Selected {rule_based_count} rule-based pages ({RULE_BASED_MODE}), {non_rule_count} padding slots")
                else:
                    print(f"[Experiment Mode 41] Selected {rule_based_count} rule-based pages ({RULE_BASED_MODE}), {non_rule_count} high-score pages")
                    
            else:
                # Default mode: top-k selection
                print(f"[Default Mode] Selecting {target_k + 1} pages, total pages: {total_page_num}")
                page_scores, selected_page_idx = selected_page_stats[:,:,:-1].topk(k=target_k, dim=-1)
                
                # Store scores for logging
                self._current_page_scores = page_scores
                self._current_page_indices = selected_page_idx
            
            # Append the last page index
            selected_page_idx = torch.cat([selected_page_idx, torch.ones_like(selected_page_idx[..., :1]) * (total_page_num - 1)], dim=-1).contiguous()
            selected_page_idx = selected_page_idx.to(torch.int32)
            
            # print(f"=== Final Selected {selected_page_idx.shape[-1]} Pages ===")
            # print(f"First 5 pages: {selected_page_idx[0, 0, :5].tolist()}")
            # print(f"Last page: {selected_page_idx[0, 0, -1].item()}")
            
            # Save selected pages to CSV file (if enabled)
            if self.enable_csv_logging:
                # Pass page scores and indices if available from experiment mode
                scores_to_save = getattr(self, '_current_page_scores', None)
                indices_to_save = getattr(self, '_current_page_indices', None)
                self._save_selected_pages_to_csv(selected_page_idx, timestep, total_page_num, 
                                               page_scores=scores_to_save, page_indices=indices_to_save)
            
            # Save page scores independently (if enabled)
            elif self.enable_score_logging:
                scores_to_save = getattr(self, '_current_page_scores', None)
                indices_to_save = getattr(self, '_current_page_indices', None)
                if scores_to_save is not None and indices_to_save is not None:
                    self._save_page_scores_to_csv(scores_to_save, indices_to_save, timestep, total_page_num)
            
            # Save to unified log file independently (if enabled)
            enable_unified_log_env = os.environ.get("ENABLE_UNIFIED_LOG", "false")
            if enable_unified_log_env.lower() == "true":
                self._save_to_unified_log(selected_page_idx, timestep)
            
            # Log emulation details for experiment mode 5
            self.log_emulation_details(selected_page_idx, timestep)

        return selected_page_idx

    @torch.no_grad()
    def forward_pure_dense(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks,
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        
        attn_output = fused_attention_pure_dense.single_query_attention(
                q,
                k,
                v,
                input_metadata.retrieval_block_tables[self.layer_idx],
                lengths_per_sample,
                self.alibi_slopes,
                self.memory_max_len,
                self.tokens_per_block,
                size_per_retrieval_token,
                timestep,
                self.rotary_embedding_dim,
                self.rotary_base,
                # self.rope_scaling_factor, # TODO: Fix rope scaling factor
                self.neox_rotary_style,
                self.kv_cache_config["INT4_ENABLED"],
                self.kv_cache_config["ZEROS_ENABLED"],
            )

        selected_page_idx = None
        return attn_output, selected_page_idx

    @torch.no_grad()
    def forward_wo_dynamic_sparse_per_tensor(
        self, 
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,  # NOTE: cached_dynamic_sparse_page_idx is of no use. Just keep for the interface consistency.
        kv_scale_quant_orig,
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)

        kv_scale_quant_orig = kv_scale_quant_orig.float()
        kv_scale_orig_quant = 1 / kv_scale_quant_orig
        
        attn_output = fused_attention_per_tensor_dense.single_query_attention(
            q,
            k,
            v,
            kv_scale_quant_orig,
            kv_scale_orig_quant,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token, 
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads,
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            2048,  # const int multiblock_switch
        )

        selected_page_idx = None
        return attn_output, selected_page_idx

    @torch.no_grad()
    def forward_w_dynamic_sparse_per_tensor(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        hidden_dim_per_retrieval_token = num_retrieval_kv_heads * self.head_dim

        if ((timestep) % self.selector_update_interval != 0) and cached_dynamic_sparse_page_idx is not None:      # Since timestep is the length of history, not including the current token. No need to -1 here.
            dynamic_sparse_page_idx = cached_dynamic_sparse_page_idx
        else:
            dynamic_sparse_page_idx = self.dynamic_select_topk_pages(    
                q, k, v,
                input_metadata.retrieval_block_tables[self.layer_idx],
                input_metadata.streaming_block_tables[self.layer_idx],
                retrieval_head_flags, head_rank_table,
                lengths_per_sample, sink_size, local_size, sink_blocks, local_blocks,
                size_per_retrieval_token, size_per_streaming_token,
                num_retrieval_kv_heads, num_streaming_kv_heads, timestep, hidden_dim_per_retrieval_token
            )
        
        kv_scale_quant_orig = kv_scale_quant_orig.float()
        kv_scale_orig_quant = 1 / kv_scale_quant_orig

        attn_output = fused_attention_per_tensor_sparse.single_query_attention(
            q,
            k,
            v,
            kv_scale_quant_orig,
            kv_scale_orig_quant,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            dynamic_sparse_page_idx,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token,
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads, 
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            self.sub_chunk_size,
            hidden_dim_per_retrieval_token,
            self.multiblock_switch,
        )

        return attn_output, dynamic_sparse_page_idx
    
    @torch.no_grad()
    def forward_wo_dynamic_sparse_fine_grained(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,        # NOTE: kv_scale_quant_orig and cached_dynamic_sparse_page_idx is of no use. Just keep for the interface consistency.
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        
        attn_output = fused_attention_fine_grained_dense.single_query_attention(
            q,
            k,
            v,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token, 
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads,
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            2048,  # const int multiblock_switch
        )

        selected_page_idx = None
        return attn_output, selected_page_idx
    
    @torch.no_grad()
    def forward_w_dynamic_sparse_fine_grained(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,        # NOTE: kv_scale_quant_orig is of no use. Just keep for the interface consistency.
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        hidden_dim_per_retrieval_token = num_retrieval_kv_heads * self.head_dim

        if ((timestep) % self.selector_update_interval != 0) and cached_dynamic_sparse_page_idx is not None:      # Since timestep is the length of history, not including the current token. No need to -1 here.
            dynamic_sparse_page_idx = cached_dynamic_sparse_page_idx
        else:
            dynamic_sparse_page_idx = self.dynamic_select_topk_pages(    
                q, k, v,
                input_metadata.retrieval_block_tables[self.layer_idx],
                input_metadata.streaming_block_tables[self.layer_idx],
                retrieval_head_flags, head_rank_table,
                lengths_per_sample, sink_size, local_size, sink_blocks, local_blocks,
                size_per_retrieval_token, size_per_streaming_token,
                num_retrieval_kv_heads, num_streaming_kv_heads, timestep, hidden_dim_per_retrieval_token
            )

        attn_output = fused_attention_fine_grained_sparse.single_query_attention(
            q,
            k,
            v,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            dynamic_sparse_page_idx,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token,
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads, 
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            self.sub_chunk_size,
            hidden_dim_per_retrieval_token,
            self.multiblock_switch,
        )

        return attn_output, dynamic_sparse_page_idx
    
    @torch.no_grad()
    def _save_selected_pages_to_csv(self, selected_page_idx, timestep, total_page_num, page_scores=None, page_indices=None):
        """
        Save selected pages information to CSV file.
        Args:
            selected_page_idx: [batch_size, num_heads, num_selected_pages] tensor
            timestep: current timestep
            total_page_num: total number of pages available
            page_scores: [batch_size, num_heads, total_pages] tensor of page scores (optional)
            page_indices: [batch_size, num_heads, total_pages] tensor of page indices sorted by score (optional)
        """
        if not self.enable_csv_logging:
            print(f"[DEBUG] CSV logging disabled for layer {self.layer_idx}, skipping save")
            return
            
        print(f"[DEBUG] Attempting to save CSV for layer {self.layer_idx}, timestep {timestep}")
        try:
            batch_size, num_heads, num_selected = selected_page_idx.shape
            
            # Create CSV header if file doesn't exist
            write_header = not os.path.exists(self.csv_filename)
            
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                if write_header:
                    if EXPERIMENT_MODE == 5:
                        header = ['timestamp', 'layer_idx', 'experiment_mode', 'timestep', 'total_pages', 
                                 'batch_idx', 'head_idx', 'selected_pages', 'num_odd_pages', 'num_even_pages',
                                 'emulation_32_pages', 'emulation_128_pages', 'full_pages', 'effective_tokens']
                    else:
                        header = ['timestamp', 'layer_idx', 'experiment_mode', 'timestep', 'total_pages', 
                                 'batch_idx', 'head_idx', 'selected_pages', 'num_odd_pages', 'num_even_pages']
                    writer.writerow(header)
                
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                for b in range(batch_size):
                    for h in range(num_heads):
                        selected_pages = selected_page_idx[b, h, :].cpu().numpy().tolist()
                        
                        # Count odd and even pages
                        odd_count = sum(1 for page in selected_pages if page % 2 == 1)
                        even_count = sum(1 for page in selected_pages if page % 2 == 0)
                        
                        # Convert selected pages list to string for CSV storage
                        pages_str = ','.join(map(str, selected_pages))
                        
                        row = [current_time, self.layer_idx, EXPERIMENT_MODE, timestep, 
                               total_page_num, b, h, pages_str, odd_count, even_count]
                        
                        # Add experiment mode 5 specific information
                        if EXPERIMENT_MODE == 5 and hasattr(self, 'emulation_masks'):
                            mask_info = self.emulation_masks.get((b, h), {})
                            emulation_32_pages = []
                            emulation_128_pages = []
                            full_pages = []
                            
                            for page_idx in selected_pages:
                                page_idx = int(page_idx)
                                masks = mask_info.get(page_idx, [2])
                                
                                for mask_type in masks:
                                    if mask_type == 0:
                                        emulation_32_pages.append(f"{page_idx}(first_half)")
                                    elif mask_type == 1:
                                        emulation_32_pages.append(f"{page_idx}(second_half)")
                                    elif mask_type == 2:
                                        if hasattr(self, '_selected_128_pairs') and page_idx in [p for score, p, pairs in self._selected_128_pairs]:
                                            emulation_128_pages.append(page_idx)
                                        else:
                                            full_pages.append(page_idx)
                            
                            # Calculate effective tokens
                            effective_tokens = len(emulation_32_pages) * 32 + len(emulation_128_pages) * 64 + len(full_pages) * 64
                            
                            row.extend([
                                ';'.join(emulation_32_pages),
                                ';'.join(map(str, emulation_128_pages)),
                                ';'.join(map(str, full_pages)),
                                effective_tokens
                            ])
                        
                        writer.writerow(row)
            
            # Save page scores to separate CSV file if available and score logging is enabled
            if page_scores is not None and page_indices is not None and self.enable_score_logging:
                self._save_page_scores_to_csv(page_scores, page_indices, timestep, total_page_num)
                        
            if timestep % 100 == 0:  # Print occasionally to avoid spam
                print(f"[CSV Log] Saved page selection for layer {self.layer_idx}, timestep {timestep} to {self.csv_filename}")
                
        except Exception as e:
            print(f"[CSV Log Error] Failed to save page selection: {e}")

    def _save_page_scores_to_csv(self, page_scores, page_indices, timestep, total_page_num):
        """
        Save page scores information to a separate CSV file for later analysis.
        Args:
            page_scores: [batch_size, num_heads, total_pages] tensor of scores sorted by value
            page_indices: [batch_size, num_heads, total_pages] tensor of page indices sorted by score  
            timestep: current timestep
            total_page_num: total number of pages available
        """
        if not self.enable_score_logging:
            return
            
        try:
            batch_size, num_heads, num_pages = page_scores.shape
            
            # Create CSV header if file doesn't exist
            write_header = not os.path.exists(self.score_csv_filename)
            
            with open(self.score_csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                if write_header:
                    header = ['timestamp', 'layer_idx', 'experiment_mode', 'timestep', 'total_pages',
                             'batch_idx', 'head_idx', 'page_idx', 'page_score', 'score_rank']
                    writer.writerow(header)
                
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                for b in range(batch_size):
                    for h in range(num_heads):
                        # Save all page scores for complete distribution analysis
                        for rank in range(num_pages):
                            page_idx = page_indices[b, h, rank].item()
                            score = page_scores[b, h, rank].item()
                            row = [current_time, self.layer_idx, EXPERIMENT_MODE, timestep,
                                   total_page_num, b, h, page_idx, score, rank]
                            writer.writerow(row)
                
            if timestep % 100 == 0:
                print(f"[CSV Log] Saved page scores for layer {self.layer_idx}, timestep {timestep} to {self.score_csv_filename}")
                
        except Exception as e:
            print(f"[CSV Score Log Error] Failed to save page scores: {e}")
    
    @torch.no_grad()
    def apply_emulation_masks(self, attention_scores, selected_page_idx, batch_size, num_heads):
        """
        Apply emulation masks for experiment mode 5.
        For 32-token emulation: mask out either first half or second half of each page.
        For 128-token emulation: use full pages.
        
        Args:
            attention_scores: [batch_size, num_heads, query_tokens, selected_pages * tokens_per_block]
            selected_page_idx: [batch_size, num_heads, num_selected_pages]
            batch_size: int
            num_heads: int
        
        Returns:
            masked_attention_scores: attention scores with emulation masks applied
        """
        if EXPERIMENT_MODE != 5 or not hasattr(self, 'emulation_masks'):
            return attention_scores
        
        # Apply masks to attention scores
        masked_attention_scores = attention_scores.clone()
        
        for b in range(batch_size):
            for h in range(num_heads):
                mask_info = self.emulation_masks.get((b, h), {})
                selected_pages = selected_page_idx[b, h].cpu().numpy()
                
                for page_idx_pos, page_idx in enumerate(selected_pages):
                    page_idx = int(page_idx)
                    masks = mask_info.get(page_idx, [2])  # Default to full page usage
                    
                    # Calculate attention score indices for this page
                    start_idx = page_idx_pos * self.tokens_per_block
                    end_idx = start_idx + self.tokens_per_block
                    
                    if end_idx > masked_attention_scores.shape[-1]:
                        continue
                    
                    # Apply mask based on emulation type
                    for mask_type in masks:
                        if mask_type == 0:  # First half only (32-token emulation)
                            # Mask out second half
                            mask_start = start_idx + self.tokens_per_block // 2
                            mask_end = end_idx
                            masked_attention_scores[b, h, :, mask_start:mask_end] = float('-inf')
                            
                        elif mask_type == 1:  # Second half only (32-token emulation)
                            # Mask out first half
                            mask_start = start_idx
                            mask_end = start_idx + self.tokens_per_block // 2
                            masked_attention_scores[b, h, :, mask_start:mask_end] = float('-inf')
                            
                        # mask_type == 2: Use full page (128-token emulation or default)
                        # No masking needed
        
        return masked_attention_scores
    
    @torch.no_grad()
    def log_emulation_details(self, selected_page_idx, timestep):
        """
        Log detailed emulation information for experiment mode 5.
        """
        if EXPERIMENT_MODE != 5 or not hasattr(self, 'emulation_masks'):
            return
            
        print(f"\n=== Experiment Mode 5 Emulation Details (Layer {self.layer_idx}, Timestep {timestep}) ===")
        
        for b in range(selected_page_idx.shape[0]):
            for h in range(selected_page_idx.shape[1]):
                mask_info = self.emulation_masks.get((b, h), {})
                selected_pages = selected_page_idx[b, h].cpu().numpy()
                
                print(f"Batch {b}, Head {h}:")
                print(f"  Selected pages: {selected_pages}")
                
                emulation_32_pages = []
                emulation_128_pages = []
                full_pages = []
                
                for page_idx in selected_pages:
                    page_idx = int(page_idx)
                    masks = mask_info.get(page_idx, [2])
                    
                    for mask_type in masks:
                        if mask_type == 0:
                            emulation_32_pages.append(f"{page_idx}(first_half)")
                        elif mask_type == 1:
                            emulation_32_pages.append(f"{page_idx}(second_half)")
                        elif mask_type == 2:
                            if page_idx in [p for score, p, pairs in getattr(self, '_selected_128_pairs', [])]:
                                emulation_128_pages.append(page_idx)
                            else:
                                full_pages.append(page_idx)
                
                print(f"  32-token emulation: {emulation_32_pages}")
                print(f"  128-token emulation: {emulation_128_pages}")  
                print(f"  Full pages: {full_pages}")
                
                # Calculate effective token coverage
                effective_32_tokens = len(emulation_32_pages) * 32
                effective_128_tokens = len(emulation_128_pages) * 64  # 128-token pages use full 64-token pages
                effective_full_tokens = len(full_pages) * 64
                total_effective_tokens = effective_32_tokens + effective_128_tokens + effective_full_tokens
                
                print(f"  Effective token coverage: {total_effective_tokens} tokens")
                print(f"    - 32-token emulation: {effective_32_tokens} tokens")
                print(f"    - 128-token emulation: {effective_128_tokens} tokens")
                print(f"    - Full pages: {effective_full_tokens} tokens")
                
        print("=" * 80)
    
    @torch.no_grad()
    def generate_emulation_mask_tensor(self, selected_page_idx):
        """
        Generate emulation mask tensor for CUDA backend.
        Returns a tensor indicating which tokens within each page should be masked.
        
        Args:
            selected_page_idx: [batch_size, num_heads, num_selected_pages]
            
        Returns:
            emulation_mask: [batch_size, num_heads, num_selected_pages, tokens_per_block]
                            1.0 for tokens to use, 0.0 for tokens to mask
        """
        if EXPERIMENT_MODE != 5 or not hasattr(self, 'emulation_masks'):
            # Return all ones (no masking) for other experiment modes
            batch_size, num_heads, num_selected = selected_page_idx.shape
            return torch.ones(batch_size, num_heads, num_selected, self.tokens_per_block, 
                            device=selected_page_idx.device, dtype=torch.float32)
        
        batch_size, num_heads, num_selected = selected_page_idx.shape
        emulation_mask = torch.ones(batch_size, num_heads, num_selected, self.tokens_per_block, 
                                   device=selected_page_idx.device, dtype=torch.float32)
        
        for b in range(batch_size):
            for h in range(num_heads):
                mask_info = self.emulation_masks.get((b, h), {})
                selected_pages = selected_page_idx[b, h].cpu().numpy()
                
                for page_pos, page_idx in enumerate(selected_pages):
                    page_idx = int(page_idx)
                    masks = mask_info.get(page_idx, [2])  # Default to full page usage
                    
                    # Apply most restrictive mask if multiple masks exist
                    for mask_type in masks:
                        if mask_type == 0:  # First half only (32-token emulation)
                            # Mask out second half
                            half_point = self.tokens_per_block // 2
                            emulation_mask[b, h, page_pos, half_point:] = 0.0
                            
                        elif mask_type == 1:  # Second half only (32-token emulation)
                            # Mask out first half
                            half_point = self.tokens_per_block // 2
                            emulation_mask[b, h, page_pos, :half_point] = 0.0
                            
                        # mask_type == 2: Use full page (128-token emulation or default)
                        # Keep all tokens (no masking)
        
        return emulation_mask
    
    @torch.no_grad()
    def _save_to_unified_log(self, selected_page_idx, timestep):
        """
        Save selected pages to unified log file in the original format.
        Each row contains all heads' selected pages as comma-separated lists.
        Args:
            selected_page_idx: [batch_size, num_heads, num_selected_pages] tensor
            timestep: current timestep
        """
        if DecodingAttentionWrapper._unified_log_file is None:
            return
            
        try:
            batch_size, num_heads, num_selected = selected_page_idx.shape
            
            # Use thread lock to ensure thread-safe writing
            with DecodingAttentionWrapper._unified_log_lock:
                with open(DecodingAttentionWrapper._unified_log_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # For each batch (usually just 1)
                    for b in range(batch_size):
                        # Create a row with all heads' selected pages
                        row_data = []
                        for h in range(num_heads):
                            selected_pages = selected_page_idx[b, h, :].cpu().numpy().tolist()
                            # Convert to string representation like in the original format
                            pages_str = str(selected_pages)
                            row_data.append(pages_str)
                        
                        # Write the row (all heads for this batch/timestep/layer)
                        writer.writerow(row_data)
                        
            if timestep % 50 == 0:  # Print occasionally
                print(f"[Unified Log] Layer {self.layer_idx}, timestep {timestep} saved to unified file")
                
        except Exception as e:
            print(f"[Unified Log Error] Failed to save to unified log: {e}")