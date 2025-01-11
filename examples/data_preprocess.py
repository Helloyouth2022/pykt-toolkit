import os, sys

# 这是为了不通过python 的 pip下载到包管理器中。而是直接通过使用pykt-toolkit文件夹下的pykt包
# os.path.dirname(__file__) 得到的是当前程序启动文件的所在的目录的路径
pykt_toolkit_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # ...xxx/xxx/pykt-toolkit
pykt_dir_path = os.path.join(pykt_toolkit_dir_path, 'pykt')  # ...xxx/xxx/pykt-toolkit/pykt

sys.path.append(pykt_dir_path)
sys.path.append(pykt_toolkit_dir_path)

import argparse
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question
from pykt.preprocess import data_proprocess, process_raw_data

# 字典 dname2paths包含不同数据集的路径，其中key是数据集的名称，value是对应数据集的路径
dname2paths = {
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

dname2paths = {
    k : pykt_toolkit_dir_path + v[2:]  for k, v in dname2paths.items()  # 把原来的.. 改为绝对路径
}

# 配置文件 data_config.json 的路径
configf = "../configs/data_config.json"
configf = pykt_toolkit_dir_path + configf[2:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="assist2015")
    parser.add_argument("-f","--file_path", type=str, default="../data/peiyou/grade3_students_b_200.csv")
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    is_debug = True
    if is_debug:  # 若是直接运行这个脚本，则可以自定义参数（不从终端启动脚本）
        args.dataset_name = "assist2009"
        args.file_path = "../data/peiyou/grade3_students_b_200.csv"
        args.min_seq_len = 3
        args.maxlen = 200
        args.kfold = 5
        # args.mode = "concept"

    print(args)

    # process raw data
    if args.dataset_name=="peiyou":
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

