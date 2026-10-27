"""Ensure the dataset ablation CLI actually forwards its V2 configuration."""
from pathlib import Path
import pytest


@pytest.mark.parametrize('variant,mode', [('akt','off'),('recent','recent'),('exact','retrieval')])
def test_v2_options_reach_training_process(tmp_path, monkeypatch, variant, mode):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import evaluate_long_mem_topk_akt_ablation as ablation
    from wandb_long_mem_topk_akt_train import parser
    p=parser()
    p.add_argument('--variants')
    p.add_argument('--seeds')
    args=p.parse_args(['--output',str(tmp_path/'run'),'--akt_source',str(tmp_path/'baseline'),
                      '--variants',variant,'--seeds','42','--dataset_name','assist2015',
                      '--architecture_variant','v2_last_retriever_residual',
                      '--training_stage','last_layer','--memory_decay_mode','log_additive',
                      '--gate_lr','0.002'])
    class Captured(Exception):
        pass
    def capture(command, **kwargs):
        expected={'architecture_variant':'v2_last_retriever_residual','training_stage':'last_layer',
                  'memory_decay_mode':'log_additive','gate_lr':'0.002','memory_mode':mode,
                  'dataset_name':'assist2015'}
        for key,value in expected.items():
            assert command[command.index('--'+key)+1]==value
        raise Captured
    monkeypatch.setattr(ablation.subprocess,'run',capture)
    with pytest.raises(Captured):
        ablation.main(args)
