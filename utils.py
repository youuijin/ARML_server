
import glob, os
import pandas as pd
import numpy as np
import random, torch


def fix_seed(seed):
    # seed = 222
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_str_path(args):
    if args.loss== "R-MAML-AT":
        sum_str_path = f"{args.imgsz}/{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.alpha}"
    elif args.loss == "trades" or args.loss =="R-MAML-trades":
        sum_str_path = f"{args.imgsz}/{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.beta}"
    elif args.loss == "WAR":
        sum_str_path = f"{args.imgsz}/{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.zeta}"
    else:
        sum_str_path = f"{args.imgsz}/{args.loss}/{args.meta_lr}"
    return sum_str_path

def add_idx(str_path, args):
    path_list = str_path.split("/")
    paths = glob.glob(f"./models/{path_list[0]}/*")
    model_paths = [(os.path.basename(i))[:-4] for i in paths]
    print(model_paths)
    idx = 0
    for path in model_paths:
        if path.startswith(path_list[1]+"_"+path_list[2]):
            idx += 1
    print(idx)

    if args.loss != "R-MAML-AT":
        args.alpha = "-"
    if args.loss != "trades" or args.loss !="R-MAML-trades":
        args.beta = "-"
    if args.loss != "WAR":
        args.zeta = "-"

    columns = ['idx(seed)', 'model', 'attack', 'eps', 'loss', 'alpha', 'beta', 'zeta', 'mela_lr', 'scheduler', 'scheduler_args', 'task_num', 'imgsz']
    datas = [idx, args.model, args.attack, args.eps, args.loss, args.alpha, args.beta, args.zeta, args.meta_lr, '-', '-', args.task_num, args.imgsz]
    if idx == 0:
        df = pd.DataFrame([datas], columns=columns)
        df.to_csv(f"./logs/train_log/{args.loss}.csv", index=False, mode='w')
    else:
        df = pd.DataFrame([datas])
        df.to_csv(f"./logs/train_log/{args.loss}.csv", index=False, header=False, mode='a')
    return idx