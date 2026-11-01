from pathlib import Path

import json

import pytest
import torch
import torch.nn.functional as F

from pykt.models.akt import Architecture
from pykt.models.yousician_akt import YousicianAKT
from pykt.models.yousician_akt_runner import evaluate_field_model, load_yousician_checkpoint
from pykt.models.train_model import model_forward
from torch.utils.data import DataLoader
from pykt.preprocess.yousician_field_preprocess import read_students, FIELD_NAMES
from pykt.preprocess.student_sequences import prepare_student_sequences, fit_encoding
from pykt.datasets.student_block_dataset import (
    StudentBlockDataset, init_student_loaders, encode_student,
    StudentBlockCache, iter_student_blocks,
)
from pykt.preprocess.yousician_preprocess_kc_fmkc_standard import _events_to_event_rows


def run_epoch(model, students, ids, block_size, cache_blocks, optimizer=None):
    """Exercise the shared pyKT batch adapter with an in-memory test fixture."""
    if model.seq_len is not None and block_size * cache_blocks < model.seq_len:
        raise ValueError("Cache capacity must cover seq_len")
    loader = DataLoader(StudentBlockDataset(students, ids, block_size, cache_blocks), batch_size=None)
    if optimizer is not None:
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            model_forward(model, batch).backward()
            optimizer.step()
    return evaluate_field_model(model, loader)


def config(drift="response"):
    return dict(field_sizes=[8] * 4, n_pid=5, d_model=16, n_blocks=2,
                dropout=0., field_dim=4, d_ff=24, num_attn_heads=2,
                final_fc_dim=16, drift=drift)


def inputs():
    torch.manual_seed(3)
    return torch.randint(1, 8, (2, 7, 4)), torch.randint(0, 2, (2, 7)), torch.randint(0, 6, (2, 7))


@pytest.mark.parametrize("drift", ["none", "shared", "response"])
def test_causality_and_padding(drift):
    model = YousicianAKT(**config(drift)).eval()
    assert isinstance(model.model, Architecture)
    if drift != "none":
        # Nonzero difficulty tests the drift branches, not just their zero init.
        with torch.no_grad():
            model.question_scalar.weight[1:].fill_(.4)
    attrs, response, pid = inputs()
    pred, reg = model(attrs, response, pid)
    assert pred.shape == response.shape
    for t in range(7):
        changed = response.clone()
        changed[:, t:] = 1 - changed[:, t:]
        torch.testing.assert_close(model(attrs, changed, pid)[0][:, :t + 1], pred[:, :t + 1])
        future = attrs.clone()
        future[:, t + 1:] = 1
        torch.testing.assert_close(model(future, response, pid)[0][:, :t + 1], pred[:, :t + 1])
    padded = model(F.pad(attrs, (0, 0, 0, 3), value=-1),
                   F.pad(response, (0, 3), value=-1), F.pad(pid, (0, 3), value=-1))
    torch.testing.assert_close(padded[0][:, :7], pred)
    torch.testing.assert_close(padded[1], reg)
    torch.testing.assert_close(F.binary_cross_entropy(padded[0][:, :7], response.float()),
                               F.binary_cross_entropy(pred, response.float()))
    assert not hasattr(model, "q_embed")
    assert hasattr(model, "wrong_drift_head") == (drift == "response")
    assert hasattr(model, "question_scalar") == (drift != "none")


def test_unknown_scalar_and_checkpoint(tmp_path):
    model = YousicianAKT(**config()).eval()
    attrs, response, pid = inputs()
    pid.zero_()
    reference = model(attrs, response, pid)
    with torch.no_grad():
        model.question_scalar.weight[0].fill_(100)
    actual = model(attrs, response, pid)
    torch.testing.assert_close(actual[0], reference[0])
    assert actual[1] == 0
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"model_config": config(), "model_state": model.state_dict(),
                "encoding": {"test": "retained"}}, checkpoint)
    restored, meta = load_yousician_checkpoint(checkpoint)
    torch.testing.assert_close(restored(attrs, response, pid)[0], reference[0])
    assert meta["encoding"] == {"test": "retained"}


def test_encoding_and_student_splits(tmp_path):
    students = {str(i): [([str(i), "inf", "0^0^0^0", "0"], str(i), i % 2)] for i in range(10)}
    prepare_student_sequences(students, FIELD_NAMES, tmp_path, kfold=3)
    artifact = json.loads((tmp_path / "student_fold_0.json").read_text())
    splits = artifact["splits"]
    assert not set(splits["train"]) & set(splits["test"] + splits["valid"])
    encoding = fit_encoding(students, splits["train"], FIELD_NAMES)
    uid = splits["test"][0]
    unseen = encode_student(students[uid], encoding)
    assert unseen["attrs"][0, 0] == 1 and unseen["pid"][0] == 0
    new_combo = encode_student([([None, "inf", "0^0^0^0", "0"], "unseen", 1)], encoding)
    assert new_combo["attrs"][0, 0] == 2
    model = YousicianAKT(**dict(config(), field_sizes=[len(v) + 3 for v in encoding["fields"]],
                                n_pid=len(encoding["questions"])))
    assert torch.isfinite(model(new_combo["attrs"][None], new_combo["response"][None], new_combo["pid"][None])[0]).all()


def test_preprocessing_identity_and_original_indices(tmp_path):
    events = json.dumps(dict(duration=[1]*4, pitches=[[69], [69], [70], [71]],
                             strings=[[0], [9], [0], [0]], reject_reason=[0, 1, 1, 0]))
    old = _events_to_event_rows(events)
    indexed = _events_to_event_rows(events, include_event_index=True)
    assert [x[0] for x in indexed] == [0, 2, 3]
    assert [x[1:] for x in indexed] == old
    assert old[0][1].split("|") == ["inf", "inf", "0^0^0^0", "0"]
    records = [dict(user_id="u", song_id="song", exercise_id="ex", days_since_signup=day,
                    play_mode="play", events_data=events) for day in [2, 1]]
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(records))
    students, skipped = read_students(path)
    assert skipped == 0 and len(students["u"]) == 6
    assert students["u"][0][1] == students["u"][3][1]
    assert students["u"][1][1] == "|".join(students["u"][1][0])
    assert students["u"][3][0][:2] == ["inf", "inf"]


def test_cache_coverage_boundaries_and_full_prefix_equivalence():
    model = YousicianAKT(**config()).eval()
    attrs, response, pid = inputs()
    events = {"attrs": attrs[0], "response": response[0], "pid": pid[0]}
    full = model(attrs[:1], response[:1], pid[:1])[0][0]
    predictions = []
    for block, offset in iter_student_blocks("a", events, 2, 4):
        pred = model(block["attrs"][None], block["response"][None], block["pid"][None])[0][0]
        predictions.append(pred[offset:])
    torch.testing.assert_close(torch.cat(predictions), full)
    limited = list(iter_student_blocks("a", events, 2, 1))
    assert [offset for _, offset in limited] == [0, 2, 2, 2]
    assert sum(len(block["response"]) - offset for block, offset in limited) == 7
    assert all(offset == 0 for _, offset in iter_student_blocks("a", events, 2, 0))
    cache = StudentBlockCache("a", 2)
    cache.append("a", events)
    with pytest.raises(ValueError):
        cache.prepend("b", events)
    with pytest.raises(ValueError):
        cache.append("b", events)
    result = run_epoch(model, {"a": events, "b": events}, ["a", "b"], 2, 1)
    assert result["events"] == 14


@pytest.mark.parametrize("drift", ["none", "shared", "response"])
def test_training_step(drift):
    model = YousicianAKT(**config(drift))
    attrs, response, pid = inputs()
    events = {"attrs": attrs[0], "response": response[0], "pid": pid[0]}
    before = model.base_head.weight.detach().clone()
    result = run_epoch(model, {"u": events}, ["u"], 3, 1,
                       optimizer=torch.optim.Adam(model.parameters(), lr=.001))
    assert result["events"] == 7
    assert torch.isfinite(torch.tensor(result["bce"]))
    assert not torch.equal(before, model.base_head.weight)


def test_original_akt_device_and_factory():
    from pykt.models.akt import AKT
    from pykt.models.init_model import init_model
    original = AKT(8, 5, 16, 2, 0., num_attn_heads=2).cpu().eval()
    _, response, pid = inputs()
    assert original(torch.ones_like(response), response, pid)[0].shape == response.shape
    cfg = config()
    del cfg["field_sizes"], cfg["n_pid"]
    model = init_model("yousician_akt", cfg, {"field_sizes": [8]*4, "num_q": 5}, "qid")
    assert isinstance(model, YousicianAKT)


def test_unseen_combination_and_cache_regularization():
    students = {"train": [(["a", "b", "c", "d"], "q1", 0),
                           (["e", "f", "g", "h"], "q2", 1)]}
    encoding = fit_encoding(students, ["train"], FIELD_NAMES)
    events = encode_student([(["a", "f", "c", "h"], "new", 1)], encoding)
    assert (events["attrs"] >= 3).all()  # Known fields, previously unseen tuple.
    model = YousicianAKT(**dict(config(), field_sizes=[5]*4, n_pid=2)).eval()
    assert torch.isfinite(model(events["attrs"][None], events["response"][None], events["pid"][None])[0]).all()
    with torch.no_grad():
        model.question_scalar.weight[1:].fill_(.5)
    selected = torch.tensor([[False, False, True]])
    _, reg = model(torch.full((1, 3, 4), 3), torch.zeros(1, 3, dtype=torch.long),
                   torch.ones(1, 3, dtype=torch.long), loss_mask=selected)
    torch.testing.assert_close(reg, torch.tensor(.25 * model.l2))


@pytest.mark.parametrize("window", [1, 3])
def test_sliding_mask_all_layers(window):
    model = YousicianAKT(**config(), seq_len=window).eval()
    attrs, response, pid = inputs()
    observed = []
    handles = []
    layers = list(model.model.blocks_1) + list(model.model.blocks_2)
    for layer in layers:
        def capture(module, args, kwargs):
            observed.append((kwargs["mask"].detach().clone(), kwargs["zero_pad"]))
        handles.append(layer.masked_attn_head.register_forward_pre_hook(capture, with_kwargs=True))
    model(attrs, response, pid)
    for handle in handles:
        handle.remove()
    assert len(observed) == 6
    for mask, strict in observed:
        for t in range(response.shape[1]):
            keys = mask[0, 0, t].nonzero().flatten().tolist()
            expected = list(range(max(0, t - window), t)) if strict else list(range(max(0, t - window + 1), t + 1))
            assert keys == expected


def test_window_causality_history_gradients_and_checkpoint(tmp_path):
    model = YousicianAKT(**config(), seq_len=3).eval()
    attrs, response, pid = inputs()
    pid.fill_(1)
    outputs = []
    handle = model.field_embeddings[0].register_forward_hook(
        lambda module, args, result: (result.retain_grad(), outputs.append(result)) and None)
    pred, _ = model(attrs, response, pid)
    pred[:, 4].sum().backward()
    assert outputs[0].grad[:, 1:4].abs().sum() > 0
    handle.remove()
    changed = response.clone()
    changed[:, 4:] = 1 - changed[:, 4:]
    torch.testing.assert_close(model(attrs, changed, pid)[0][:, :5], pred[:, :5])
    checkpoint = tmp_path / "window.ckpt"
    torch.save({"model_config": dict(config(), seq_len=3), "model_state": model.state_dict()}, checkpoint)
    restored, _ = load_yousician_checkpoint(checkpoint)
    assert restored.seq_len == 3
    torch.testing.assert_close(restored(attrs, response, pid)[0], pred)
    events = {"attrs": attrs[0], "response": response[0], "pid": pid[0]}
    with pytest.raises(ValueError, match="capacity"):
        run_epoch(model, {"u": events}, ["u"], 2, 1)
    assert run_epoch(model, {"u": events}, ["u"], 3, 1)["events"] == 7


def test_prepared_pipeline_shared_factories_and_training(tmp_path):
    from pykt.datasets import init_dataset4train, init_test_datasets
    from pykt.models import init_model, train_model, evaluate, load_model
    students = {str(i): [([str(i), "inf", "0", "0"], str(i), r)
                         for r in (0, 1, 0, 1, 1)] for i in range(10)}
    entry = prepare_student_sequences(students, FIELD_NAMES, tmp_path, kfold=3)
    entry.update(dpath=str(tmp_path), dataset_name="fixture")
    all_config = {"fixture": entry}
    options = dict(block_size=3, cache_blocks=1, seq_len=3)
    train, valid, test = init_dataset4train("fixture", "yousician_akt", all_config,
                                          0, 1, args=options)
    assert not set(train.dataset.ids) & set(valid.dataset.ids + test.dataset.ids)
    assert entry["selected_fold"] == 0
    # Held-out field and question identities stay unknown in this fold.
    held_out = next(iter(test))
    assert (held_out["attrs"][:, 0] == 1).all()
    assert (held_out["pid"] == 0).all()
    batches = list(train)
    assert sum(int(b["loss_mask"].sum()) for b in batches) == 5 * len(train.dataset.ids)
    for uid in train.dataset.ids:
        own = [b for b in batches if b["uid"] == uid]
        assert own[0]["loss_mask"].all()
        assert not own[1]["loss_mask"][:3].any()
    cfg = config()
    del cfg["field_sizes"], cfg["n_pid"]
    cfg["seq_len"] = 3
    model = init_model("yousician_akt", cfg, entry, "qid")
    output = tmp_path / "checkpoint"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_config": cfg, "data_config": entry}))
    result = train_model(model, train, valid, 1, torch.optim.Adam(model.parameters()),
                         str(output), save_model=True)
    assert result[-1] == 1
    loaded = load_model("yousician_akt", cfg, entry, "qid", str(output))
    paired, metadata = load_yousician_checkpoint(output / "qid_model.ckpt")
    assert metadata["data_config"]["encoding"] == entry["encoding"]
    test_again, _, _, _ = init_test_datasets(entry, "yousician_akt", 1)
    assert evaluate(loaded, test_again, "yousician_akt") == evaluate(paired, test, "yousician_akt")
    # Runtime files contain prepared events only; no raw source is needed.
    assert set(json.loads((tmp_path / "student_sequences.json").read_text())) == set(students)
    for fold in entry["folds"]:
        artifact = json.loads((tmp_path / f"student_fold_{fold}.json").read_text())
        assert set(artifact["encoding"]["questions"]) == set(artifact["splits"]["train"])


def test_preprocess_then_train_without_raw_file(tmp_path):
    from pykt.preprocess.yousician_field_preprocess import prepare_dataset
    from pykt.datasets import init_dataset4train
    events = json.dumps(dict(duration=[1, 1], pitches=[[69], [70]],
                             strings=[[0], [0]], reject_reason=[0, 1]))
    source = tmp_path / "source.json"
    source.write_text(json.dumps([dict(user_id=str(i), song_id="s", exercise_id=str(i),
        days_since_signup=1, play_mode="play", events_data=events) for i in range(5)]))
    output, cfg_path = tmp_path / "prepared", tmp_path / "data_config.json"
    prepare_dataset(source, output, cfg_path, "yousician", kfold=2)
    source.unlink()
    snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
    configuration = json.loads(cfg_path.read_text())
    train, valid, test = init_dataset4train("yousician", "yousician_akt", configuration,
        1, 1, args=dict(block_size=2, cache_blocks=1, seq_len=2))
    cfg = config()
    cfg.update(field_sizes=configuration["yousician"]["field_sizes"],
               n_pid=configuration["yousician"]["num_q"], seq_len=2)
    model = YousicianAKT(**cfg)
    model_forward(model, next(iter(train))).backward()
    assert evaluate_field_model(model, test)["events"] == 2
    assert snapshot == {p.name: p.read_bytes() for p in output.iterdir()}


def test_question_identity_is_only_local_kc_vector(tmp_path):
    events = json.dumps(dict(duration=[1, 1, 1], pitches=[[69], [69], [69]],
                             strings=[[0], [0], [0]], reject_reason=[0, 1, 0]))
    records = [dict(user_id=str(i), song_id=f"song{i}", exercise_id=f"ex{i}",
                    exercise_part_index=i, days_since_signup=i,
                    play_mode="play", events_data=events) for i in range(3)]
    # Neither score identity field is necessary for preprocessing.
    records[2].pop("song_id")
    records[2].pop("exercise_id")
    path = tmp_path / "local.json"
    path.write_text(json.dumps(records))
    students, _ = read_students(path)
    assert students["0"][1][1] == students["0"][2][1]  # Different event positions.
    for uid in students:
        assert students[uid][1][1] == students["0"][1][1]
        assert all(pid == "|".join(attrs) for attrs, pid, _ in students[uid])
    assert students["0"][0][1] != students["0"][1][1]  # Different prev fields.
    encoding = fit_encoding(students, ["0"], FIELD_NAMES)
    assert len(encoding["questions"]) == 2
    for uid in students:
        encoded = encode_student(students[uid], encoding)
        assert encoded["pid"].tolist() == [1, 2, 2]


@pytest.mark.parametrize('name', ['akt', 'akt_fmkc'])
def test_comparison_models_share_blocks_and_mask_regularization(tmp_path, name):
    from pykt.models import init_model
    from pykt.datasets import init_dataset4train
    students = {str(i): [(['a', 'b', 'c', 'd'], 'a|b|c|d', r)
                         for r in (0, 1, 0, 1, 1)] for i in range(10)}
    entry = prepare_student_sequences(students, FIELD_NAMES, tmp_path, kfold=3)
    entry.update(dpath=str(tmp_path), dataset_name='fixture')
    train, valid, test = init_dataset4train('fixture', name, {'fixture': entry},
        0, 1, args=dict(block_size=3, cache_blocks=1, seq_len=3))
    cfg = config()
    del cfg['field_sizes'], cfg['n_pid']
    model = init_model(name, dict(cfg, seq_len=3), entry, 'qid')
    assert model.model_name == name
    assert model.seq_len == 3
    batch = list(train)[1]
    assert batch['loss_mask'].tolist() == [False, False, False, True, True]
    model_forward(model, batch).backward()
    assert evaluate_field_model(model, test)['events'] == 5 * len(test.dataset.ids)
    if name == 'akt':
        assert hasattr(model, 'q_embed') and not hasattr(model, 'field_embeddings')
        with torch.no_grad():
            model.difficult_param.weight.fill_(.5)
        _, reg = model(batch['pid'][None], batch['response'][None], batch['pid'][None],
                       loss_mask=batch['loss_mask'][None])
        torch.testing.assert_close(reg, torch.tensor(2 * .25 * model.l2, device=reg.device))
    else:
        assert hasattr(model, 'field_embeddings') and not hasattr(model, 'q_embed')


def test_one_click_comparison(tmp_path):
    from examples.compare_yousician_akt import main
    events = json.dumps(dict(duration=[1, 1], pitches=[[69], [70]],
                             strings=[[0], [0]], reject_reason=[0, 1]))
    source = tmp_path / 'raw.json'
    source.write_text(json.dumps([dict(user_id=str(i), days_since_signup=1,
        play_mode='play', events_data=events) for i in range(5)]))
    results = main(['--raw_file', str(source), '--save_dir', str(tmp_path / 'run'),
        '--kfold', '2', '--num_epochs', '1', '--d_model', '16', '--d_ff', '24',
        '--n_blocks', '1', '--num_attn_heads', '2', '--field_dim', '4',
        '--block_size', '2', '--seq_len', '2'])
    assert [r['model_name'] for r in results] == ['akt', 'akt_fmkc']
    configs = [json.loads((Path(r['checkpoint']).parent / 'config.json').read_text()) for r in results]
    assert configs[0]['data_config'] == configs[1]['data_config']
    assert all(Path(r['checkpoint']).exists() and r['test']['events'] == 2 for r in results)
    assert (tmp_path / 'run/comparison.json').exists()
