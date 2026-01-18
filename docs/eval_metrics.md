# Eval Metrics Map

以下为进度条/日志中的指标对照。所有 *_t 都是毫秒 (ms)。

| full_name | abbrev | meaning |
| --- | --- | --- |
| avg_time | avg_t | 所有 forward 的平均总时间。包含 vision_time、prune_llm/unprune_llm、diff_time、bg_time、region_time 等；不包含环境动作/渲染。 |
| eval_time | eval_t | 触发 region 评估的 forward 的平均总时间。 |
| noeval_time | noeval_t | 未触发 region 评估的 forward 的平均总时间。 |
| vision_time | vis_t | vision encoder 的前向时间。 |
| diff_time | diff_t | 去噪过程的时间 (action expert)，不包含 region 评估里额外的去噪。 |
| bg_time | bg_t | 背景评估时间。 |
| region_time | reg_t | region 评估时间（仅对评估帧求平均，包含扰动推理）。 |
| mask_time | mask_t | 剪枝前的 mask 构建时间（仅对不评估帧求平均）。 |
| prune_time | prn_t | 剪枝打包时间（仅对不评估帧求平均）。 |
| prune_llm | p_llm | 剪枝后 language model 的前向时间 (不含 vision 和 action expert)。 |
| unprune_llm | u_llm | 未剪枝时 language model 的前向时间 (不含 vision 和 action expert)。 |
| pruned_tokens | prn_tok | 每个不评估 forward 平均真正剪掉的 token 数量。 |
| pruned_ratio | prn_ratio | 每个不评估 forward 平均剪掉的 token 比例（基于该 forward 的总视觉 token）。 |
| bg_ratio | bg_ratio | 可剪掉的背景 token 比例与数量（**按 forward 平均**），格式 avg_count/ratio%。 |
| region_ratio | reg_ratio | 重要 region 的 token 比例与数量（**按 forward 平均**），格式 avg_count/ratio%。 |
| sr | sr | 任务成功率 (百分比)。 |
