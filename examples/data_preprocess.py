import os, sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question
from pykt.preprocess import data_proprocess, process_raw_data

dname2paths = {
    "yousician": "../data/yousician/yousician_ukulele.json",
    "yousician_fmkc": "../data/yousician_fmkc/yousician_ukulele.json",
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2012": "../data/assist2012/2012-2013-data-with-predictions-4-final.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv",
    "poj": "../data/poj/poj_log.csv",
    "slepemapy": "../data/slepemapy/answer.csv",
    "assist2017": "../data/assist2017/anonymized_full_release_competition_dataset.csv",
    "junyi2015": "../data/junyi2015/junyi_ProblemLog_original.csv",
    "ednet": "../data/ednet/",
    "ednet5w": "../data/ednet/",
    "peiyou": "../data/peiyou/grade3_students_b_200.csv"
}
configf = "../configs/data_config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="assist2015")
    parser.add_argument("-f","--file_path", type=str, default=None)
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--input_format", choices=["yousician", "vector"], default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config_file", default=configf)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    if args.model_name in {"yousician_akt", "akt_fmkc"} or (args.model_name in {None, "akt"} and
            args.dataset_name in {"yousician", "yousician_fmkc", "pianokt"}):
        input_format = args.input_format or ("yousician" if args.dataset_name in
            {"yousician", "yousician_fmkc"} else "vector")
        if input_format == "yousician":
            from pykt.preprocess.yousician_field_preprocess import prepare_dataset
        else:
            from pykt.preprocess.vector_kc_preprocess import prepare_dataset
        source = args.file_path or dname2paths.get(args.dataset_name)
        if not source:
            parser.error("Provide --file_path for this dataset")
        prepare_dataset(source, args.output_dir or "../data/" + args.dataset_name,
                        args.config_file, args.dataset_name, args.kfold, args.seed)
        sys.exit(0)

    print(args)

    # process raw data
    if args.dataset_name=="peiyou" and args.file_path is not None:
        dname2paths["peiyou"] = args.file_path
        print(f"fpath: {args.file_path}")
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-"*50)
    print(f"dname: {dname}, writef: {writef}")
    # split
    os.system("rm " + dname + "/*.pkl")

    #for concept level model
    split_concept(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
    print("="*100)

    #for question level model
    split_question(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)

