"""Compare dense AKT and vector AKT-FMKC on a shared student dataset."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.train_akt_fmkc import build_parser
from examples.wandb_train import main as train
from pykt.preprocess.yousician_field_preprocess import prepare_dataset


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(save_dir=str(ROOT / 'saved_model' / 'yousician_comparison'))
    parser.add_argument('--raw_file', default=str(ROOT / 'data/yousician/yousician_ukulele.json'))
    parser.add_argument('--input_format', choices=['yousician', 'vector'], default='yousician')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--split_seed', type=int, default=1024)
    parser.add_argument('--prepared_config', help='Reuse prepared data instead of preprocessing raw data')
    args = parser.parse_args(argv)
    if args.num_epochs < 1 or not 0 <= args.fold < args.kfold:
        parser.error('Require positive num_epochs and 0 <= fold < kfold')
    output = Path(args.save_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.prepared_config:
        config_path = Path(args.prepared_config).resolve()
    else:
        config_path = output / 'data_config.json'
        prepare = prepare_dataset
        if args.input_format == 'vector':
            from pykt.preprocess.vector_kc_preprocess import prepare_dataset as prepare
        prepare(args.raw_file, output / 'prepared', config_path,
                        args.dataset_name, args.kfold, args.split_seed)
    params = vars(args).copy()
    for key in ('raw_file', 'kfold', 'split_seed', 'prepared_config', 'input_format'):
        params.pop(key)
    params['data_config'] = str(config_path)
    results = []
    for model_name in ('akt', 'akt_fmkc'):
        result = train(dict(params, model_name=model_name, save_dir=str(output / model_name)))
        result['kc_encoding'] = 'dense_id' if model_name == 'akt' else 'field_vector'
        results.append(result)
        (output / 'comparison.json').write_text(json.dumps(results, indent=2))
    print('\nModel          Test AUC    Test ACC    Test BCE')
    for result in results:
        m = result['test']
        print(f"{result['model_name']:14s} {m['auc']:.6f}    {m['accuracy']:.6f}    {m['bce']:.6f}")
    print(f"Results: {output / 'comparison.json'}")
    return results


if __name__ == '__main__':
    main()
