"""Numerical AKT equivalence, leakage and actual multi-group training tests."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pykt.models.akt import AKT
from pykt.models.long_mem_akt import LongMemAKT
from pykt.models.long_mem_akt_runner import train_long_mem_akt, evaluate_long_mem_akt
from pykt.datasets.long_mem_akt_dataset import LongMemAKTDataset

torch.set_num_threads(1)


def make_model(**kwargs):
    torch.manual_seed(7)
    config = dict(d_model=16, n_blocks=2, num_attn_heads=4, d_ff=32,
                  final_fc_dim=32, num_slots=4, block_size=4, dropout=0.)
    config.update(kwargs)
    config.setdefault("phase", "candidate")
    model = LongMemAKT(12, 12, **config)
    if model.phase == "candidate":
        with torch.no_grad():
            model.correction_head[-1].weight.normal_(std=.05)
    return model


def block():
    c = torch.tensor([[1, 2, 3, 4]])
    return c, c.clone(), torch.tensor([[0., 1., 0., 1.]]), torch.arange(4).double()[None], torch.ones_like(c, dtype=torch.bool)


def forward(model, b, memory=None):
    c, q, r, t, valid = b
    if memory is None:
        memory = model.initial_memory(c.size(0), c.device)
    return model.predict_block(c, q, r, valid, memory, has_history=True)


@pytest.mark.parametrize("separate", [False, True])
def test_short_branch_equals_original_akt(separate):
    m = make_model(phase="base", separate_qa=separate).eval()
    original = AKT(12, 12, 16, 2, 0., d_ff=32, num_attn_heads=4,
                   final_fc_dim=32, separate_qa=separate).eval()
    with torch.no_grad():
        m.akt.difficult_param.weight.normal_()
    original.load_state_dict(m.akt.state_dict())
    c, q, r, _, valid = block()
    expected, reg = original(c, r.long(), q)
    actual, cache = forward(m, block())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cache["reg_loss"], reg)
    qe, qa, diff, _ = original.encode_embeddings(c, r.long(), q)
    torch.testing.assert_close(cache["query"], qe)
    torch.testing.assert_close(cache["interaction"], qa)
    torch.testing.assert_close(qe, original.q_embed(c) + diff * original.q_embed_diff(c))
    expected.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in original.named_parameters() if "gammas" in name)


def test_no_pid_uses_only_concept_embedding():
    m = LongMemAKT(12, 0, d_model=16, n_blocks=1, num_attn_heads=4, dropout=0., phase="base")
    c, q, r, _, _ = block()
    _, cache = forward(m, block())
    torch.testing.assert_close(cache["query"], m.akt.q_embed(c))
    torch.testing.assert_close(cache["interaction"], m.akt.q_embed(c) + m.akt.qa_embed(r.long()))


def test_every_prediction_is_causal_and_snapshot_is_immutable():
    m = make_model().eval()
    b = block()
    memory = m.initial_memory(1, "cpu")
    before = memory.clone()
    p, cache = forward(m, b, memory)
    for t in range(4):
        changed = list(copy.deepcopy(b))
        changed[2][:, t:] = 1 - changed[2][:, t:]
        changed[0][:, t+1:] = 8
        changed[1][:, t+1:] = 9
        other, _ = forward(m, changed, memory)
        torch.testing.assert_close(p[:, :t+1], other[:, :t+1])
    new = m.write(memory, m.summarize(b[2], b[4], b[3], cache))
    torch.testing.assert_close(memory, before)
    torch.testing.assert_close(forward(m, b, memory)[0], p)
    assert not torch.allclose(new, memory)


def test_write_encoding_is_causal_and_order_sensitive():
    m = make_model().eval()
    b = block()
    _, cache = forward(m, b)
    z = m.encode_write(b[4], cache)
    changed = list(copy.deepcopy(b))
    changed[2][:, 2:] = 1 - changed[2][:, 2:]
    changed[0][:, 2:] = 7
    _, cache2 = forward(m, changed)
    torch.testing.assert_close(z[:, :2], m.encode_write(b[4], cache2)[:, :2])
    same = list(copy.deepcopy(b))
    same[0].fill_(1)
    same[1].fill_(1)
    _, a = forward(m, same)
    s1 = m.summarize(same[2], b[4], None, a)
    same[2] = same[2].flip(1)
    _, a = forward(m, same)
    s2 = m.summarize(same[2], b[4], None, a)
    assert not torch.allclose(s1["summary"], s2["summary"])


def assert_nonzero_grad(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("detach", [False, True])
def test_second_block_supervises_exclusive_writer_and_encoder_parameters(detach):
    m = make_model()
    b = block()
    memory = m.initial_memory(1, "cpu")
    _, cache = forward(m, b, memory)
    summary = m.summarize(b[2], b[4], b[3], cache)
    updated = m.write(memory, summary)
    if detach:
        updated = updated.detach()
    _, next_cache = forward(m, b, updated)
    # No first-block loss or second-block write: exclusively tests historical writing.
    F.binary_cross_entropy_with_logits(next_cache["logits"], b[2]).backward()
    for module in (m.erase, m.add, m.write_encoder, m.write_z):
        if detach:
            assert all(p.grad is None for p in module.parameters())
        else:
            assert_nonzero_grad(module)


def test_delayed_boundary_write_trains_writer_not_detached_encoder():
    m = make_model()
    b = block()
    memory = m.initial_memory(1, "cpu")
    _, cache = forward(m, b, memory)
    summary = {k: v.detach() for k, v in m.summarize(b[2], b[4], b[3], cache).items()}
    updated = m.write(memory.detach(), summary)
    _, cache = forward(m, b, updated)
    F.binary_cross_entropy_with_logits(cache["logits"], b[2]).backward()
    assert_nonzero_grad(m.erase)
    assert_nonzero_grad(m.add)
    assert all(p.grad is None for p in m.write_encoder.parameters())


def test_padding_does_not_change_predictions_summary_or_memory():
    m = make_model(block_size=8).eval()
    b = block()
    memory = m.initial_memory(1, "cpu")
    pred, cache = forward(m, b, memory)
    summary = m.summarize(b[2], b[4], b[3], cache)
    padded = [F.pad(t, (2, 2), value=False if t.dtype == torch.bool else -1) for t in b]
    p2, cache2 = forward(m, padded, memory)
    s2 = m.summarize(padded[2], padded[4], padded[3], cache2)
    torch.testing.assert_close(p2[:, 2:6], pred)
    torch.testing.assert_close(s2["summary"], summary["summary"])
    torch.testing.assert_close(m.write(memory, s2), m.write(memory, summary))
    padded[4].fill_(False)
    p3, cache3 = forward(m, padded, memory)
    s3 = m.summarize(padded[2], padded[4], padded[3], cache3)
    assert torch.isfinite(p3).all() and torch.isfinite(s3["summary"]).all()
    torch.testing.assert_close(m.write(memory, s3), memory)


def test_student_storage_and_predictions_are_isolated():
    m = make_model().eval()
    a = m.initial_memory(1, "cpu")
    b = m.initial_memory(1, "cpu")
    assert a.data_ptr() != b.data_ptr() != m.memory_init.data_ptr()
    b_pred = forward(m, block(), b)[0]
    with torch.no_grad():
        a.add_(10)
    torch.testing.assert_close(forward(m, block(), b)[0], b_pred)
    torch.testing.assert_close(b, m.initial_memory(1, "cpu"))


def test_time_units_missing_time_and_disabled_weighting():
    m = make_model().eval()
    b = block()
    _, cache = forward(m, b)
    wall = m.summarize(b[2], b[4], b[3]*1000 + 1700000000000, cache)
    events = m.summarize(b[2], b[4], None, cache)
    missing = m.summarize(b[2], b[4], torch.full_like(b[3], float("nan")), cache)
    assert wall["uses_timestamps"].all() and not missing["uses_timestamps"].any()
    torch.testing.assert_close(wall["alpha"], events["alpha"])
    torch.testing.assert_close(missing["alpha"], events["alpha"])
    m.time_weight = False
    m.event_distance_fallback = True
    disabled = m.summarize(b[2], b[4], None, cache)
    torch.testing.assert_close(disabled["alpha"], torch.ones_like(b[2]))


@pytest.mark.parametrize("mode", ["short", "long", "concat", "gate", "selective_residual"])
def test_removed_modes_rejected(mode):
    with pytest.raises(TypeError):
        make_model(ablation=mode)



def dataset(tmp_path):
    path = tmp_path / "students.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "fold", "concepts", "questions", "responses"])
        writer.writeheader()
        for uid, fold, length in [("001", 0, 17), ("002", 1, 11), ("003", -1, 13)]:
            writer.writerow(dict(uid=uid, fold=fold, concepts=",".join(str(i % 10) for i in range(length)),
                                 questions=",".join(str(i % 10) for i in range(length)),
                                 responses=",".join(str(i % 2) for i in range(length))))
    return path


@pytest.mark.parametrize("group_size", [2, 3])
def test_tbptt_optimizer_steps_do_not_reuse_old_graph(tmp_path, group_size):
    data = LongMemAKTDataset(dataset(tmp_path), ["concepts", "questions"], {0}, 4, 3)
    loader = DataLoader(data, batch_size=1)
    m = make_model(tbptt_blocks=group_size)
    optimizer = torch.optim.Adam(m.parameters())
    with patch.object(m, "write", wraps=m.write) as write:
        for _ in range(2):
            assert train_long_mem_akt(m, loader, optimizer) > 0
        assert write.call_count == 8


def test_bad_configuration_is_not_silently_ignored():
    with pytest.raises(TypeError):
        make_model(typo_heads=4)
    with pytest.raises(ValueError):
        make_model(tbptt_blocks=0)


def test_akt_reset_only_zeros_pid_difficulty():
    # PID count + 1 equals d_model: old shape-based reset zeroed attention layers.
    m = AKT(15, 15, 16, 1, 0., num_attn_heads=4)
    assert m.difficult_param.weight.count_nonzero() == 0
    assert m.q_embed_diff.weight.count_nonzero() > 0
    assert m.model.blocks_1[0].masked_attn_head.k_linear.weight.count_nonzero() > 0


def test_time_free_independent_write_is_order_invariant():
    m = make_model(write_encoding="independent", time_weight=False).eval()
    b = list(block())
    b[0].fill_(1)
    b[1].fill_(1)
    _, c1 = forward(m, b)
    s1 = m.summarize(b[2], b[4], None, c1)
    b[2] = b[2].flip(1)
    _, c2 = forward(m, b)
    s2 = m.summarize(b[2], b[4], None, c2)
    torch.testing.assert_close(s1["summary"], s2["summary"])



def residual_model(**kwargs):
    m = make_model(**kwargs)
    with torch.no_grad():
        m.correction_head[-1].weight.zero_()
        m.correction_head[-1].bias.zero_()
    return m


def residual_forward(m, b=None, history=True, memory=None):
    c, q, r, _, valid = b if b is not None else block()
    return m.predict_block(c, q, r, valid, memory if memory is not None else m.initial_memory(1, "cpu"),
                           has_history=history, uid="001", block_index=1)


def test_candidate_zero_init_and_future_loss_writer_gradients_and_frozen_base():
    m = residual_model().train()
    b = block()
    _, first = residual_forward(m)
    assert torch.equal(first["logits"], first["base_logits"])
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.01)
    before = {k: p.clone() for k, p in m.akt.state_dict().items()}
    F.binary_cross_entropy_with_logits(first["logits"], b[2]).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    memory = m.initial_memory(1, "cpu")
    _, c1 = residual_forward(m, history=False, memory=memory)
    memory = m.write(memory, m.summarize(b[2], b[4], b[3], c1))
    _, c2 = residual_forward(m, memory=memory)
    F.binary_cross_entropy_with_logits(c2["logits"], b[2]).backward()
    for module in [m.erase, m.add, m.write_encoder, m.write_z]:
        assert_nonzero_grad(module)
    assert all(p.grad is None and not p.requires_grad for p in m.akt.parameters())
    assert all(torch.equal(p, before[k]) for k, p in m.akt.state_dict().items())
    assert not m.akt.training and m.write_encoder.training
    with pytest.raises(ValueError, match="tbptt_blocks"):
        residual_model(tbptt_blocks=1)




def test_e1_train_valid_test_cli_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    work = tmp_path / "examples"
    work.mkdir()
    configs = tmp_path / "configs"
    configs.mkdir()
    path = tmp_path / "students.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "fold", "concepts", "questions", "responses"])
        writer.writeheader()
        for student in range(18):
            fold = 0 if student < 12 else (1 if student < 15 else -1)
            writer.writerow(dict(uid=f"{student:03d}", fold=fold,
                                 concepts=",".join(str(i%10) for i in range(13)),
                                 questions=",".join(str(i%10) for i in range(13)),
                                 responses=",".join(str((i+student)%2) for i in range(13))))
    data = dict(dpath=str(tmp_path), input_type=["questions", "concepts"], folds=[0, 1],
                maxlen=4, min_seq_len=3, num_c=12, num_q=12,
                train_valid_original_file=path.name, test_original_file=path.name)
    (configs / "data_config.json").write_text(json.dumps({"synthetic": data}))
    (configs / "wandb.json").write_text("{}")
    env = dict(os.environ, PYTHONPATH=str(root), MPLCONFIGDIR=str(tmp_path / "mpl"))
    command = [sys.executable, str(root / "examples/wandb_long_mem_akt_train.py"),
               "--stage", "train", "--output", str(tmp_path / "run"), "--block_size", "4",
               "--dataset_name", "synthetic", "--fold", "1", "--epochs", "2",
               "--threads", "1", "--d_model", "16", "--n_blocks", "1",
               "--num_attn_heads", "4", "--d_ff", "32", "--final_fc_dim", "32"]
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = tmp_path / "run"
    assert (checkpoint / "qid_model.ckpt").exists()
    result = subprocess.run([sys.executable, str(root / "examples/wandb_long_mem_akt_train.py"),
                             "--stage", "evaluate", "--source", str(checkpoint), "--block_size", "4",
                             "--output", str(tmp_path / "evaluation")],
                            cwd=work, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert report["count"] == 27
    result = subprocess.run([sys.executable, str(root / "examples/wandb_predict.py"),
                             "--save_dir", str(checkpoint), "--use_wandb", "0"],
                            cwd=work, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    standard = json.loads((checkpoint / "test_metrics.json").read_text())
    assert standard["brier"] == pytest.approx(report["brier"])
    from pykt.models.long_mem_akt_workflow import load_artifact, role_loader
    model, _, _, roles = load_artifact(checkpoint)
    assert set(roles["roles"]) == {"train", "valid", "test"}
    assert len(roles["roles"]["train"]) == 12
    assert len(roles["roles"]["valid"]) == 3
    assert not any("benefit" in key or "calibration" in key for key in model.state_dict())
    model.set_stage("inference")
    _, _, evaluated = evaluate_long_mem_akt(model, role_loader(roles, "test"))
    assert evaluated["brier"] == pytest.approx(report["brier"])
    result = subprocess.run([sys.executable, str(root / "examples/evaluate_long_mem_akt_ablation.py"),
                             "--source", str(checkpoint), "--output", str(tmp_path / "comparison"),
                             "--threads", "1"], cwd=work, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    comparison = json.loads((tmp_path / "comparison/metrics.json").read_text())
    assert set(comparison) == {"E0", "E1"}
    assert comparison["E1"]["brier"] == pytest.approx(report["brier"])


def test_full_residual_inference_without_selector():
    m = residual_model().eval()
    with torch.no_grad():
        m.correction_head[-1].bias.fill_(.8)
    _, before = residual_forward(m)
    m.set_stage("inference")
    _, after = residual_forward(m)
    torch.testing.assert_close(before["logits"], after["logits"])
    torch.testing.assert_close(after["logits"], after["base_logits"] + .8)
    _, first = residual_forward(m, history=False)
    assert torch.equal(first["logits"], first["base_logits"])
    b = list(block())
    for t in range(4):
        changed = copy.deepcopy(b)
        changed[2][:, t:] = 1 - changed[2][:, t:]
        changed[0][:, t+1:] = 9
        _, cache = residual_forward(m, changed)
        torch.testing.assert_close(after["logits"][:, :t+1], cache["logits"][:, :t+1])


@pytest.mark.parametrize("option", ["mc_samples", "mc_seed", "benefit_hidden"])
def test_removed_selector_options_rejected(option):
    with pytest.raises(TypeError):
        make_model(**{option: 4})


def test_e1_nonfinite_fallback_and_empty_memory_write():
    m = residual_model().eval()
    with torch.no_grad():
        m.correction_head[-1].bias.fill_(float('nan'))
    _, cache = residual_forward(m)
    assert torch.equal(cache['logits'], cache['base_logits'])
    assert (cache['fallback_reason'] == 2).all()
    b = list(block())
    b[4].fill_(False)
    memory = m.initial_memory(1, 'cpu')
    probability, cache = residual_forward(m, b, memory=memory)
    assert torch.isfinite(probability).all()
    summary = m.summarize(b[2], b[4], b[3], cache)
    torch.testing.assert_close(m.write(memory, summary), memory)


def test_train_valid_test_roles_and_artifact_integrity(tmp_path):
    from pykt.models.long_mem_akt_workflow import validate_roles, save_artifact, load_artifact
    roles = dict(train=['a'], valid=['b'], test=['c'])
    validate_roles(roles)
    with pytest.raises(ValueError):
        validate_roles(dict(roles, test=['a']))
    mc = dict(d_model=16, n_blocks=1, num_attn_heads=4, block_size=4, dropout=0.)
    model = LongMemAKT(12, 12, **mc).eval()
    manifest = dict(roles=roles, fold=0, data_config=dict(num_c=12, num_q=12, dataset_name='synthetic'))
    save_artifact(tmp_path, model, mc, manifest, 'train')
    restored, _, _, _ = load_artifact(tmp_path)
    restored.eval()
    torch.testing.assert_close(residual_forward(model)[0], residual_forward(restored)[0])
    config = json.loads((tmp_path / 'config.json').read_text())
    config['model_config']['block_size'] = 8
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='checksum'):
        load_artifact(tmp_path)
