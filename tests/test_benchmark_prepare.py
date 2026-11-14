import json
from unittest.mock import patch

import pytest

from pykt.benchmark.prepare import prepare
from pykt.datasets.student_block_dataset import init_student_loaders


def synthetic():
    return {str(i): [(['inf', 'inf', str(i), '0'], f'inf|inf|{i}|0', j % 2)
                    for j in range(5)] for i in range(20)}


def test_nested_training_only_and_selective_loading(tmp_path):
    raw = tmp_path / 'raw.json'
    raw.write_text('[]')
    root = tmp_path / 'run'
    students = synthetic()
    with patch('pykt.benchmark.prepare.read_students', return_value=(students, 0)):
        tasks = prepare(raw, root, sizes=(2, 4, 'full'))
    assert len(tasks) == 6
    previous = set()
    holds = []
    for task in tasks[::2]:
        config = json.loads(open(task['prepared_config_path']).read())['yousician']
        loaders = init_student_loaders(config, 0, block_size=2, cache_blocks=1, seq_len=2)
        split = config['splits']
        assert previous <= set(split['train'])
        previous = set(split['train'])
        holds.append((split['valid'], split['test']))
        assert set(config['encoding']['questions']) == {
            kc for uid in split['train'] for _, kc, _ in students[uid]}
        assert set(loaders[0].dataset.students) == set(sum(split.values(), []))
        for loader, name in zip(loaders, ('train', 'valid', 'test')):
            assert sum(int(batch['loss_mask'].sum()) for batch in loader) == 5 * len(split[name])
    assert holds[0] == holds[1] == holds[2]
    with pytest.raises(FileExistsError):
        prepare(raw, root)


def test_unavailable_size_has_no_partial_output(tmp_path):
    raw = tmp_path / 'raw.json'
    raw.write_text('[]')
    with patch('pykt.benchmark.prepare.read_students', return_value=(synthetic(), 0)):
        with pytest.raises(ValueError, match='unavailable'):
            prepare(raw, tmp_path / 'run', sizes=(240,))
    assert not (tmp_path / 'run').exists()
