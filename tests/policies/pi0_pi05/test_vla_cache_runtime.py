#!/usr/bin/env python

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def _load_functions(module_relpath: str):
    module_path = ROOT / module_relpath
    source = module_path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(module_path))
    wanted = {"compute_prefix_layer_full", "compute_prefix_layer_with_reuse"}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    compiled = compile(ast.Module(body=functions, type_ignores=[]), str(module_path), "exec")

    def _gated_residual(residual, update, gate):  # noqa: ARG001
        return residual + update

    def _apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1):  # noqa: ARG001
        return query_states, key_states

    def _eager_attention_forward(self_attn, query_states, key_states, value_states, attention_mask, scaling):
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * scaling
        scores = scores + attention_mask
        attn_weights = torch.softmax(scores, dim=-1)
        att_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return att_output, attn_weights

    namespace = {
        "torch": torch,
        "modeling_gemma": SimpleNamespace(
            _gated_residual=_gated_residual,
            apply_rotary_pos_emb=_apply_rotary_pos_emb,
            eager_attention_forward=_eager_attention_forward,
        ),
    }
    exec(compiled, namespace)
    return namespace["compute_prefix_layer_full"], namespace["compute_prefix_layer_with_reuse"]


class CountingLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        self.tokens_seen = 0

    def forward(self, inputs):
        self.tokens_seen += int(inputs.shape[1])
        return super().forward(inputs)


class FakeNorm(torch.nn.Module):
    def forward(self, hidden_states, cond=None):  # noqa: ARG002
        return hidden_states, torch.zeros_like(hidden_states)


class FakeMLP(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.up_proj = CountingLinear(width, width)

    def forward(self, inputs):
        return self.up_proj(inputs)


class FakeSelfAttention(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.num_heads = 1
        self.head_dim = width
        self.scaling = width ** -0.5
        self.q_proj = CountingLinear(width, width)
        self.k_proj = CountingLinear(width, width)
        self.v_proj = CountingLinear(width, width)
        self.o_proj = CountingLinear(width, width)


class FakeLayer(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.input_layernorm = FakeNorm()
        self.post_attention_layernorm = FakeNorm()
        self.self_attn = FakeSelfAttention(width)
        self.mlp = FakeMLP(width)


def _rotary_emb(dummy_tensor, position_ids):  # noqa: ARG001
    return None, None


def _reset_counters(layer: FakeLayer) -> None:
    layer.self_attn.q_proj.tokens_seen = 0
    layer.self_attn.k_proj.tokens_seen = 0
    layer.self_attn.v_proj.tokens_seen = 0
    layer.self_attn.o_proj.tokens_seen = 0
    layer.mlp.up_proj.tokens_seen = 0


@pytest.mark.parametrize(
    "module_relpath",
    [
        "src/lerobot/policies/pi0/modeling_pi0.py",
        "src/lerobot/policies/pi05/modeling_pi05.py",
    ],
)
def test_compute_prefix_layer_with_reuse_only_processes_non_reused_tokens(module_relpath: str):
    compute_prefix_layer_full, compute_prefix_layer_with_reuse = _load_functions(module_relpath)
    torch.manual_seed(0)
    layer = FakeLayer(width=4)
    hidden_states = torch.randn(1, 4, 4)
    pad_masks = torch.tensor([[True, True, True, True]])
    reuse_mask = torch.tensor([[False, True, False, True]])
    attention_mask = torch.zeros(1, 1, 4, 4)
    position_ids = torch.arange(4)[None]

    full_outputs, full_keys, full_values, full_attn = compute_prefix_layer_full(
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        _rotary_emb,
        return_attentions=True,
    )

    _reset_counters(layer)
    outputs, key_states, value_states, attn_weights = compute_prefix_layer_with_reuse(
        layer=layer,
        hidden_states=hidden_states,
        pad_masks=pad_masks,
        attention_mask=attention_mask,
        position_ids=position_ids,
        rotary_emb=_rotary_emb,
        reuse_mask=reuse_mask,
        prev_layer_output=full_outputs,
        prev_key_states=full_keys,
        prev_value_states=full_values,
        prev_attn_weights=full_attn,
        return_attentions=True,
    )

    assert torch.allclose(outputs, full_outputs, atol=1e-6)
    assert torch.allclose(key_states, full_keys, atol=1e-6)
    assert torch.allclose(value_states, full_values, atol=1e-6)
    assert torch.allclose(attn_weights, full_attn, atol=1e-6)
    assert layer.self_attn.q_proj.tokens_seen == 2
    assert layer.self_attn.k_proj.tokens_seen == 2
    assert layer.self_attn.v_proj.tokens_seen == 2
    assert layer.self_attn.o_proj.tokens_seen == 2
    assert layer.mlp.up_proj.tokens_seen == 2


@pytest.mark.parametrize(
    "module_relpath",
    [
        "src/lerobot/policies/pi0/modeling_pi0.py",
        "src/lerobot/policies/pi05/modeling_pi05.py",
    ],
)
def test_compute_prefix_layer_with_reuse_preserves_previous_attention_rows(module_relpath: str):
    compute_prefix_layer_full, compute_prefix_layer_with_reuse = _load_functions(module_relpath)
    torch.manual_seed(1)
    layer = FakeLayer(width=4)
    hidden_states = torch.randn(1, 4, 4)
    pad_masks = torch.tensor([[True, True, True, True]])
    reuse_mask = torch.tensor([[True, False, True, False]])
    attention_mask = torch.zeros(1, 1, 4, 4)
    position_ids = torch.arange(4)[None]

    full_outputs, full_keys, full_values, full_attn = compute_prefix_layer_full(
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        _rotary_emb,
        return_attentions=True,
    )
    prev_attn = torch.full_like(full_attn, 0.25)

    _, _, _, attn_weights = compute_prefix_layer_with_reuse(
        layer=layer,
        hidden_states=hidden_states,
        pad_masks=pad_masks,
        attention_mask=attention_mask,
        position_ids=position_ids,
        rotary_emb=_rotary_emb,
        reuse_mask=reuse_mask,
        prev_layer_output=full_outputs,
        prev_key_states=full_keys,
        prev_value_states=full_values,
        prev_attn_weights=prev_attn,
        return_attentions=True,
    )

    reused_idx = torch.tensor([0, 2])
    recomputed_idx = torch.tensor([1, 3])
    assert torch.allclose(attn_weights[:, :, reused_idx, :], prev_attn[:, :, reused_idx, :], atol=1e-6)
    assert torch.allclose(attn_weights[:, :, recomputed_idx, :], full_attn[:, :, recomputed_idx, :], atol=1e-6)


def test_token_selection_state_resets_cuda_latency_lists():
    module_path = ROOT / "src/lerobot/policies/token_selection_utils.py"
    spec = importlib.util.spec_from_file_location("token_selection_utils_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    state = module.TokenSelectionState(
        cuda_latency_ms=[1.0, 2.0],
        cuda_latency_with_reuse_ms=[1.5],
        cuda_latency_without_reuse_ms=[2.5],
    )
    state.reset()

    assert state.cuda_latency_ms == []
    assert state.cuda_latency_with_reuse_ms == []
    assert state.cuda_latency_without_reuse_ms == []
