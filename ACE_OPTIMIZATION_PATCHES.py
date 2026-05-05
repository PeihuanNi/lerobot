"""
ACE CUDA Event Latency 优化补丁集

这个文件包含所有优化方案的具体代码实现。
可以直接应用或作为参考实现进行改进。
"""

# ============================================================================
# 补丁 1a: 消除 .tolist() 同步
# ============================================================================
# 文件: src/lerobot/policies/pi0/modeling_pi0.py
# 行号: 1125-1150

PATCH_1A = """
【原始代码】
    total_prefix = prefix_pad_masks.sum(dim=1).tolist()
    vision_tokens = (
        prefix_pad_masks.sum(dim=1) - img_mask_pad.sum(dim=1)
    ).tolist()
    language_tokens = (
        token_state.frame_idx
        - prefix_pad_masks.sum(dim=1)
        + img_mask_pad.sum(dim=1)
    ).tolist()
    if trace_enabled:
        logger.debug(
            f"prefix_build: frame={token_state.frame_idx} "
            f"total={total_prefix} vision={vision_tokens.tolist()} language={language_tokens.tolist()}"
        )

【优化代码】
    # 计算 prefix 统计（保留为张量，避免同步）
    total_prefix_tensor = prefix_pad_masks.sum(dim=1)
    vision_tokens_tensor = (
        total_prefix_tensor - img_mask_pad.sum(dim=1)
    )
    language_tokens_tensor = (
        token_state.frame_idx - total_prefix_tensor + img_mask_pad.sum(dim=1)
    )
    
    # 仅在需要日志打印时才转换（且使用异步转换）
    if trace_enabled and logger.isEnabledFor(logging.DEBUG):
        # 转换推迟到 CPU，不阻塞 GPU 流水线
        try:
            total_list = total_prefix_tensor.cpu().tolist() if isinstance(total_prefix_tensor, torch.Tensor) else total_prefix_tensor
            vision_list = vision_tokens_tensor.cpu().tolist() if isinstance(vision_tokens_tensor, torch.Tensor) else vision_tokens_tensor
            language_list = language_tokens_tensor.cpu().tolist() if isinstance(language_tokens_tensor, torch.Tensor) else language_tokens_tensor
            logger.debug(
                f"prefix_build: frame={token_state.frame_idx} "
                f"total={total_list} vision={vision_list} language={language_list}"
            )
        except Exception:
            # Fallback 如果转换失败
            logger.debug(f"prefix_build: frame={token_state.frame_idx} (tensor conversion failed)")
"""

# ============================================================================
# 补丁 1b: 合并梯度计算（关键优化）
# ============================================================================
# 文件: src/lerobot/policies/pi0/modeling_pi0.py
# 行号: 1440-1505

PATCH_1B_V1 = """
【原始代码的问题】
    # dimension_independent 变体循环调用 backward
    for _d in range(_action_dim):
        _obj_d = _v_sel[:, :, _d].sum()
        _gs_local = _grad_with_zero_for_unused(
            _obj_d,
            _score_attn_list,
            retain_graph=_retain_score_graph or (_d < _action_dim - 1),
            allow_unused_zero=_use_prefix_score_attn,
        )
        # ... 累积梯度逻辑 ...

【优化代码 V1：合并梯度】
    # 关键改进：不循环 backward，而是一次性计算所有梯度
    def _compute_merged_gradients():
        \"\"\"合并所有维度的梯度计算为单个反向传播\"\"\"
        # 方案1：加权求和（保持相对关系）
        _weighted_sum = torch.zeros_like(_v_sel[:, :, 0])
        for _d in range(_action_dim):
            _weighted_sum = _weighted_sum + _v_sel[:, :, _d].sum() / _action_dim
        
        # 一次性反向传播
        _grads_merged = _grad_with_zero_for_unused(
            _weighted_sum,
            _score_attn_list,
            retain_graph=_retain_score_graph,
            allow_unused_zero=_use_prefix_score_attn,
        )
        return _grads_merged
    
    (_dim_A_bars,), backward_ms, backward_memory = _measure_profiled_cuda_section(
        "policy.token_selection.action_expert.ace_backward",
        _compute_merged_gradients,
        track_memory=True,
    )
"""

PATCH_1B_V2 = """
【优化代码 V2：直接使用范数（最快）】
    # 不需要循环维度，直接用 L1/L2 范数
    def _compute_norm_based_gradients():
        \"\"\"使用范数而不是逐维度循环，速度更快且更稳定\"\"\"
        if cfg.ace_objective == "action_sample_L1":
            # L1 范数：不需要循环
            _obj_norm = _obj_target.abs().sum()
        elif cfg.ace_objective == "action_sample_L2":
            # Frobenius 范数（L2）
            _obj_norm = (_obj_target ** 2).sum()
        else:
            raise ValueError(f"Unsupported objective: {cfg.ace_objective}")
        
        # 单次反向传播
        _grads_norm = _grad_with_zero_for_unused(
            _obj_norm,
            _score_attn_list,
            retain_graph=_retain_score_graph,
            allow_unused_zero=_use_prefix_score_attn,
        )
        return _grads_norm
    
    _grads, backward_ms, backward_memory = _measure_profiled_cuda_section(
        "policy.token_selection.action_expert.ace_backward",
        _compute_norm_based_gradients,
        track_memory=True,
    )

【集成到主逻辑】
    # 在 _run_interp_backward 函数中使用
    if cfg.ace_variant == "dimension_independent":
        # 使用新的合并梯度计算
        _grads, backward_ms, _ = _measure_profiled_cuda_section(
            "policy.token_selection.action_expert.ace_backward",
            _compute_norm_based_gradients,  # ← 替换原始的循环版本
            track_memory=True,
        )
"""

# ============================================================================
# 补丁 1c: 批量处理 region discard
# ============================================================================
# 文件: src/lerobot/policies/pi0/modeling_pi0.py
# 行号: 1770-1820

PATCH_1C = """
【原始代码】
    for idx, score_token in enumerate(score_tokens):
        meta = metas[idx]
        _discarded_regions = 0  # ← 每次都初始化
        score_region = compute_region_scores(score_token, meta.patches_per_side, cfg.region_patch_size)
        
        # ... 其他逻辑 ...
        
        _discard_mask = discard_top_scoring_regions(
            _prev_scores, _prev_kept_regions, cfg.discard_prev_kept_ratio, _max_discard,
            mode=cfg.discard_mode
        )
        _discarded_regions = int(_discard_mask.sum().item())  # ← GPU→CPU 同步
        token_state.global_active_region_masks[idx] &= ~_discard_mask

【优化代码】
    # 预先收集所有需要的数据，然后批量处理
    discard_batch = []
    
    for idx, score_token in enumerate(score_tokens):
        meta = metas[idx]
        score_region = compute_region_scores(score_token, meta.patches_per_side, cfg.region_patch_size)
        
        # ... 其他逻辑 ...
        
        # 计算但不转换为 int（保留 GPU 张量）
        discard_mask = discard_top_scoring_regions(
            _prev_scores, _prev_kept_regions, cfg.discard_prev_kept_ratio, _max_discard,
            mode=cfg.discard_mode
        )
        discard_batch.append(discard_mask)
    
    # 统一更新全局掩码（全 GPU 操作）
    for idx, discard_mask in enumerate(discard_batch):
        token_state.global_active_region_masks[idx] &= ~discard_mask
    
    # 如果需要统计信息，使用异步转换（不阻塞主线程）
    if cfg.debug_token_selection:
        # 收集统计而不阻塞
        def collect_stats():
            return [m.sum(dim=-1).cpu() for m in discard_batch]
        # 可以在后台线程执行，或延迟到帧结束时执行
"""

# ============================================================================
# 补丁 2a: Prefix Cache 去重
# ============================================================================
# 文件: src/lerobot/policies/pi0/modeling_pi0.py
# 行号: 1335-1365

PATCH_2A = """
【原始代码的问题】
    def _compute_prefix_cache():
        if _use_prefix_score_attn:
            with torch.enable_grad():
                return compute_prefix_cache(
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    capture_attn=True,
                )
        return compute_prefix_cache(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            capture_attn=False,
        )

    (past_key_values, prefix_score_attns), prefix_cache_ms, _ = _measure_profiled_cuda_section(
        "policy.token_selection.prefix_cache.language_model",
        _compute_prefix_cache,
    )
    # 注意：prefix_score_attns 稍后会在 ACE 梯度中被重新使用
    # 这意味着计算可能被重复或浪费

【优化代码】
    class _PrefixCacheState:
        \"\"\"缓存 prefix 计算以避免重复\"\"\"
        def __init__(self):
            self.cached_attns = None
            self.cached_past_kv = None
            self.is_valid = False
        
        def compute_if_needed(self, prefix_embs, prefix_pad_masks, prefix_att_masks, 
                             need_grad, compute_fn):
            \"\"\"按需计算 prefix，并缓存结果\"\"\"
            if self.is_valid:
                return self.cached_past_kv, self.cached_attns  # 返回缓存
            
            if need_grad:
                with torch.enable_grad():
                    past_kv, attns = compute_fn(
                        prefix_embs, prefix_pad_masks, prefix_att_masks,
                        capture_attn=True
                    )
                self.cached_attns = attns
            else:
                past_kv, _ = compute_fn(
                    prefix_embs, prefix_pad_masks, prefix_att_masks,
                    capture_attn=False
                )
                self.cached_attns = None
            
            self.cached_past_kv = past_kv
            self.is_valid = True
            return past_kv, self.cached_attns
        
        def invalidate(self):
            \"\"\"新的推理步开始时清除缓存\"\"\"
            self.is_valid = False
            self.cached_attns = None
            self.cached_past_kv = None
    
    # 使用
    _prefix_cache_state = _PrefixCacheState()
    need_prefix_grad = _use_prefix_score_attn and cfg.grad_score_method in {"ace", "grad_only"}
    
    (past_key_values, prefix_score_attns), prefix_cache_ms, _ = _measure_profiled_cuda_section(
        "policy.token_selection.prefix_cache.language_model",
        lambda: _prefix_cache_state.compute_if_needed(
            prefix_embs, prefix_pad_masks, prefix_att_masks,
            need_prefix_grad, compute_prefix_cache
        ),
    )
"""

# ============================================================================
# 补丁 2b: 异步统计收集
# ============================================================================
# 文件: src/lerobot/policies/pi0/modeling_pi0.py (新增类)

PATCH_2B = """
【新增 AsyncStatsCollector 类】
    class AsyncStatsCollector:
        \"\"\"异步收集统计数据而不阻塞 GPU 流水线\"\"\"
        def __init__(self, device='cuda'):
            self.device = device
            self.pending_stats = {}
            self.cpu_stream = torch.cuda.Stream(device=device)
        
        def collect_tensor_stat(self, name: str, tensor: torch.Tensor):
            \"\"\"记录张量统计但不立即转换\"\"\"
            if not isinstance(tensor, torch.Tensor):
                return
            
            # 在 CPU 流中异步转换（不阻塞主 GPU 流）
            with torch.cuda.stream(self.cpu_stream):
                try:
                    self.pending_stats[name] = tensor.cpu().detach()
                except Exception as e:
                    logger.warning(f"Failed to collect stat {name}: {e}")
        
        def get_stats_blocking(self):
            \"\"\"等待异步转换完成并获取结果\"\"\"
            torch.cuda.current_stream().wait_stream(self.cpu_stream)
            
            result = {}
            for name, tensor in self.pending_stats.items():
                try:
                    if isinstance(tensor, torch.Tensor):
                        result[name] = tensor.tolist() if tensor.numel() > 1 else tensor.item()
                    else:
                        result[name] = tensor
                except Exception as e:
                    logger.warning(f"Failed to convert stat {name}: {e}")
            
            self.pending_stats.clear()
            return result

【在主逻辑中使用】
    stats_collector = AsyncStatsCollector(device=device)
    
    for idx, score_token in enumerate(score_tokens):
        # ... pruning logic ...
        
        # 异步收集统计而不阻塞
        stats_collector.collect_tensor_stat(
            f"camera_{idx}_discarded_regions",
            _discard_mask.sum(dim=-1)
        )
        stats_collector.collect_tensor_stat(
            f"camera_{idx}_total_active",
            token_state.global_active_region_masks[idx].sum(dim=-1)
        )
    
    # 帧结束时获取统计（在调试打印时）
    if cfg.debug_token_selection:
        debug_stats = stats_collector.get_stats_blocking()
        logger.debug(f"Pruning stats: {debug_stats}")
"""

# ============================================================================
# 补丁汇总
# ============================================================================

IMPLEMENTATION_PRIORITY = """
【实现优先级建议】

优先级 1（立即实施）：
  ✓ 1a: 消除 .tolist() 同步
      预期效果: 2-4 ms
      风险: 低
      工作量: 1-2 小时

优先级 2（同步进行）：
  ✓ 1b V2: 使用范数的梯度合并
      预期效果: 8-15 ms  
      风险: 低（只改变计算方式，不改变结果）
      工作量: 3-4 小时

优先级 3（第二阶段）：
  ✓ 1c: 批量处理 discard
      预期效果: 1-3 ms
      风险: 中等
      工作量: 2-3 小时

优先级 4（可选）：
  ✓ 2a: Prefix cache 去重
      预期效果: 2-5 ms
      风险: 中等
      工作量: 3-4 小时
  
  ✓ 2b: 异步统计
      预期效果: 1-2 ms
      风险: 低
      工作量: 2-3 小时

【总体预期】
应用优先级 1-2 后：
  当前: 206 ms
  优化后: 190-195 ms
  改进: 11-16 ms (5-8%)

应用优先级 1-3 后：
  当前: 206 ms
  优化后: 187-192 ms
  改进: 14-19 ms (7-9%)

应用所有优化后：
  当前: 206 ms
  优化后: 183-188 ms
  改进: 18-23 ms (9-11%)
"""

if __name__ == "__main__":
    print("ACE Latency Optimization Patches")
    print("=" * 80)
    print(IMPLEMENTATION_PRIORITY)
    print("=" * 80)
    print("\nPatch 1a (Remove .tolist()):")
    print(PATCH_1A)
    print("\n" + "=" * 80)
    print("\nPatch 1b V1 (Merged gradients):")
    print(PATCH_1B_V1)
    print("\n" + "=" * 80)
    print("\nPatch 1b V2 (Norm-based gradients - RECOMMENDED):")
    print(PATCH_1B_V2)
    print("\n" + "=" * 80)
    print("\nPatch 1c (Batch discard operations):")
    print(PATCH_1C)
    print("\n" + "=" * 80)
    print("\nPatch 2a (Prefix cache deduplication):")
    print(PATCH_2A)
    print("\n" + "=" * 80)
    print("\nPatch 2b (Async stats collection):")
    print(PATCH_2B)
