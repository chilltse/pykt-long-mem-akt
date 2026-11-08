import copy
import json
from pathlib import Path

import pytest
import torch

from pykt.preprocess.vector_kc_preprocess import read_students, prepare_dataset, category
from pykt.datasets.student_block_dataset import init_student_loaders
from pykt.models.init_model import init_model
from pykt.models.akt_fmkc import AKTFMKC
from pykt.models.akt_fmkc_runner import field_batch_loss, load_fmkc_checkpoint

SOURCE = Path(__file__).parent / 'fixtures/pianokt_grouped.json'


def test_piano_example():
    students, names = read_students(SOURCE)
    assert len(names) == 6
    events = next(iter(students.values()))
    assert len(events) == 6
    assert [e[2] for e in events] == [0] * 6
    assert events[0][0][3] is None
    assert len({category(v) for v in [False, 0, '0', True, 1, '1']}) == 6


@pytest.mark.parametrize('width', [1, 6, 9])
def test_variable_width_causality(width):
    model = AKTFMKC([6]*width, 2, d_model=8, n_blocks=1, dropout=0,
                    field_dim=3, num_attn_heads=2, seq_len=3).eval()
    attrs = torch.randint(0, 6, (1, 5, width))
    responses = torch.tensor([[0, 1, 0, 1, 0]])
    pid = torch.ones_like(responses)
    pred = model(attrs, responses, pid)[0]
    changed = responses.clone()
    changed[:, 2:] = 1 - changed[:, 2:]
    torch.testing.assert_close(pred[:, :3], model(attrs, changed, pid)[0][:, :3])
    with pytest.raises(ValueError, match='shape'):
        model(torch.zeros(1, 5, width+1, dtype=torch.long), responses, pid)


def test_six_field_preprocess_train_checkpoint(tmp_path):
    record = json.loads(SOURCE.read_text())
    records = copy.deepcopy(record)
    records['students'] = []
    for i in range(5):
        r = copy.deepcopy(record['students'][0])
        r['learner_key'] = str(i)
        records['students'].append(r)
    source = tmp_path / 'attempts.json'
    source.write_text(json.dumps(records))
    cfgpath = tmp_path / 'config.json'
    prepare_dataset(source, tmp_path / 'prepared', cfgpath, kfold=4)
    config = json.loads(cfgpath.read_text())['pianokt']
    train, valid, test = init_student_loaders(config, 0, 2, 1, 2)
    assert len(config['field_sizes']) == 6
    assert not set(train.dataset.ids) & set(valid.dataset.ids + test.dataset.ids)
    assert (next(iter(train))['attrs'][0, [3, 5]] == 2).all()
    mc = dict(d_model=8, n_blocks=1, dropout=0., field_dim=3,
              num_attn_heads=2, seq_len=2)
    model = init_model('akt_fmkc', mc, config, 'qid')
    optimizer = torch.optim.Adam(model.parameters())
    for batch in train:
        optimizer.zero_grad()
        field_batch_loss(model, batch).backward()
        optimizer.step()
    (tmp_path / 'config.json').write_text(json.dumps(dict(model_config=mc, data_config=config)))
    checkpoint = tmp_path / 'model.ckpt'
    torch.save(model.state_dict(), checkpoint)
    restored, _ = load_fmkc_checkpoint(checkpoint, next(model.parameters()).device)
    model.eval()
    batch = next(iter(test))
    torch.testing.assert_close(field_batch_loss(model, batch), field_batch_loss(restored, batch))


def test_bad_schema_and_insufficient_students(tmp_path):
    with pytest.raises(ValueError, match='students'):
        prepare_dataset(SOURCE, tmp_path / 'prepared', tmp_path / 'config.json', kfold=2)
    assert not (tmp_path / 'prepared').exists()
    record = json.loads(SOURCE.read_text())
    record['students'][0]['visits'][0]['attempts'][0]['events_data']['kc_vectors'][1].pop()
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='width'):
        read_students(path)


def test_grouped_ordering_and_labels(tmp_path):
    record = json.loads(SOURCE.read_text())
    expected, names = read_students(SOURCE)
    for student in record['students']:
        student['visits'].reverse()
        for visit in student['visits']:
            visit['attempts'].reverse()
            for attempt in visit['attempts']:
                events = attempt['events_data']
                for key in ('kc_vectors', 'correct', 'onset_seconds'):
                    events[key].reverse()
                # Test correct-only exports as well as reject_reason exports.
                del events['reject_reason']
    path = tmp_path / 'shuffled.json'
    path.write_text(json.dumps(record))
    assert read_students(path) == (expected, names)
    events['correct'][0] = 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='binary'):
        read_students(path)


def test_five_students_keep_original_split_policy(tmp_path):
    record = json.loads(SOURCE.read_text())
    student = record['students'][0]
    record['students'] = [dict(copy.deepcopy(student), learner_key=str(i)) for i in range(5)]
    path = tmp_path / 'five.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Not enough students'):
        prepare_dataset(path, tmp_path / 'prepared', tmp_path / 'config.json')
    entry = prepare_dataset(path, tmp_path / 'prepared', tmp_path / 'config.json', kfold=4)
    assert entry['folds'] == [0, 1, 2, 3]
    for fold in entry['folds']:
        artifact = json.loads((tmp_path / 'prepared' / f'student_fold_{fold}.json').read_text())
        assert {key: len(ids) for key, ids in artifact['splits'].items()} == {
            'train': 3, 'valid': 1, 'test': 1}


def test_conflicting_labels(tmp_path):
    record = json.loads(SOURCE.read_text())
    record['students'][0]['visits'][0]['attempts'][0]['events_data']['correct'][0] = 1
    path = tmp_path / 'conflicting.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='disagree'):
        read_students(path)
