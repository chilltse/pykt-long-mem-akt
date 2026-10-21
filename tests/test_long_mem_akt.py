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
    return LongMemAKT(12, 12, **config)


def block():
    c = torch.tensor([[1, 2, 3, 4]])
    return c, c.clone(), torch.tensor([[0., 1., 0., 1.]]), torch.arange(4).double()[None], torch.ones_like(c, dtype=torch.bool)


def forward(model, b, memory=None):
    c, q, r, t, valid = b
    if memory is None:
        memory = model.initial_memory(c.size(0), c.device)
    return model.predict_block(c, q, r, valid, memory)


@pytest.mark.parametrize("separate", [False, True])
def test_short_branch_equals_original_akt(separate):
    m = make_model(ablation="short", separate_qa=separate).eval()
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
    m = LongMemAKT(12, 0, d_model=16, n_blocks=1, num_attn_heads=4, dropout=0.)
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


@pytest.mark.parametrize("mode", ["short", "long", "concat", "gate"])
def test_branch_ablations(mode):
    m = make_model(ablation=mode)
    _, cache = forward(m, block())
    F.binary_cross_entropy_with_logits(cache["logits"], block()[2]).backward()
    if mode == "concat":
        assert m.concat_head[0].in_features == 3 * 16
        assert_nonzero_grad(m.concat_head)
    if mode in ("long", "concat", "gate"):
        assert m.memory_init.grad is not None and m.memory_init.grad.abs().sum() > 0
    if mode == "short":
        assert m.memory_init.grad is None


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


def test_real_data_training_multiple_groups_checkpoint_and_student_eval(tmp_path):
    from pykt.datasets import init_dataset4train, init_test_datasets
    from pykt.models import init_model, train_model, load_model
    path = dataset(tmp_path)
    config = dict(dpath=str(tmp_path), dataset_name="synthetic", input_type=["questions", "concepts"],
                  folds=[0, 1], maxlen=4, min_seq_len=3, num_c=12, num_q=12,
                  train_valid_original_file=path.name, test_original_file=path.name)
    train, valid = init_dataset4train("synthetic", "long_mem_akt", {"synthetic": config}, 1, 8)
    test, window, _, _ = init_test_datasets(config, "long_mem_akt", 8)
    assert window is None
    assert train.dataset[0]["uid"] == "001" and valid.dataset[0]["uid"] == "002"
    assert test.dataset[0]["uid"] == "003"
    assert train.dataset[0]["smasks"].sum() == 12
    config_model = dict(d_model=16, n_blocks=1, num_attn_heads=4, d_ff=32,
                        final_fc_dim=32, block_size=4, tbptt_blocks=2, dropout=0.)
    m = init_model("long_mem_akt", config_model, config, "qid")
    optimizer = torch.optim.Adam(m.parameters(), lr=.001)
    before = m.memory_init.detach().clone()
    with patch.object(m, "write", wraps=m.write) as write:
        loss = train_long_mem_akt(m, train, optimizer)
        assert loss > 0 and write.call_count == 4
    assert not torch.equal(before, m.memory_init)
    results = train_model(m, train, valid, 2, optimizer, str(tmp_path), save_model=True)
    assert len(results) == 7 and (tmp_path / "qid_model.ckpt").exists()
    restored = load_model("long_mem_akt", config_model, config, "qid", str(tmp_path))
    output = tmp_path / "predictions.csv"
    auc, acc, diagnostics = evaluate_long_mem_akt(restored, test, str(output), student_id="003")
    assert 0 <= auc <= 1 and 0 <= acc <= 1 and diagnostics["count"] == 9
    with output.open() as f:
        assert len(list(csv.DictReader(f))) == 13
    restored.score_protocol = "student"
    assert evaluate_long_mem_akt(restored, test)[2]["count"] == 12


@pytest.mark.parametrize("group_size", [1, 2, 3])
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


def test_cli_train_and_predict_one_student(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = dataset(tmp_path)
    work = tmp_path / "examples"
    work.mkdir()
    configs = tmp_path / "configs"
    configs.mkdir()
    data = dict(dpath=str(tmp_path), input_type=["questions", "concepts"], folds=[0, 1],
                maxlen=4, min_seq_len=3, num_c=12, num_q=12,
                train_valid_original_file=path.name, test_original_file=path.name)
    (configs / "data_config.json").write_text(json.dumps({"synthetic": data}))
    (configs / "kt_config.json").write_text(json.dumps({"train_config": dict(
        batch_size=1, num_epochs=1, optimizer="adam", seq_len=4)}))
    (configs / "wandb.json").write_text("{}")
    env = dict(os.environ, PYTHONPATH=str(root), MPLCONFIGDIR=str(tmp_path / "mpl"))
    result = subprocess.run([sys.executable, str(root / "examples/wandb_long_mem_akt_train.py"),
        "--dataset_name", "synthetic", "--fold", "1", "--save_dir", str(tmp_path / "models"),
        "--num_epochs", "1", "--block_size", "4", "--d_model", "16", "--n_blocks", "1",
        "--num_attn_heads", "4", "--d_ff", "32", "--final_fc_dim", "32"],
        cwd=work, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = next((tmp_path / "models").glob("*/qid_model.ckpt"))
    result = subprocess.run([sys.executable, str(root / "examples/wandb_predict.py"),
        "--save_dir", str(checkpoint.parent), "--use_wandb", "0", "--student_id", "003"],
        cwd=work, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Student-ordered nonoverlapping blocks" in result.stdout
    assert (checkpoint.parent / "student_block_predictions.csv").exists()
