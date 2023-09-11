
import glob, os
import pandas as pd
import numpy as np
import random, torch
from datetime import datetime

def fix_seed(seed=222):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_str_path(args):
    if args.loss.startswith("AT"):
        sum_str_path = f"{args.imgsz}/{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.alpha}"
    elif args.loss.startswith("trades"):
        sum_str_path = f"{args.imgsz}/{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.beta}"
    else:
        sum_str_path = f"{args.imgsz}/{args.loss}/{args.meta_lr}"

    if args.loss.find("WAR") >= 0:
        sum_str_path += f"_{args.zeta}"

    return sum_str_path

def add_idx(str_path, args):
    # path_list = str_path.split("/")
    # paths = glob.glob(f"./models/{path_list[0]}/*")
    # model_paths = [(os.path.basename(i))[:-4] for i in paths]

    # idx = 0
    # for path in model_paths:
    #     if path.startswith(path_list[1]+"_"+path_list[2]):
    #         idx += 1

    alpha, beta, zeta = args.alpha, args.beta, args.zeta
    

    if args.loss != "AT" and args.loss != "AT-WAR":
        alpha = "-"
    if args.loss != "trades" and args.loss != "trades-WAR":
        beta = "-"
    if args.loss != "AT-WAR" and args.loss != "trades-WAR":
        zeta = "-"

    now = datetime.now()
    date = now.strftime('%Y-%m-%d %H:%M:%S')
    columns = ['loss', 'alpha', 'beta', 'zeta', 'attack', 'eps', 'model', 'meta_lr', 'scheduler', 'scheduler_args', 'task_num', 'imgsz', 'date']
    datas = [args.loss, alpha, beta, zeta, args.attack, args.eps, args.model, args.meta_lr, args.scheduler, scheduler_args, args.task_num, args.imgsz, date]
    df = pd.DataFrame([datas], columns=columns)
    # df.to_csv(f"./logs/train_log.csv", index=True, header=True, mode='w') # to make new csv file
    df.to_csv(f"./logs/train_log.csv", index=True, header=False, mode='a')
    #return idx

def add_log(args):
    alpha, beta, zeta = args.alpha, args.beta, args.zeta
    
    if args.loss != "AT" and args.loss != "AT-WAR":
        alpha = "-"
    if args.loss != "trades" and args.loss != "trades-WAR":
        beta = "-"
    if args.loss != "AT-WAR" and args.loss != "trades-WAR":
        zeta = "-"

    scheduler_args = ""
    if args.sche == "lambda":
        scheduler_args = f"lambda : {args.sche_arg1}"
    elif args.sche == "step":
        scheduler_args = f"step: {args.sche_arg1} gamma: {args.sche_arg2}"
    elif args.sche == "cosine":
        scheduler_args = f"T max: {args.sche_arg1} eta-min: {args.sche_arg2}"
        
    now = datetime.now()
    date = now.strftime('%Y-%m-%d %H:%M:%S')
    columns = ['loss', 'alpha', 'beta', 'zeta', 'attack', 'eps', 'model', 'meta_lr', 'scheduler', 'scheduler_args', 'task_num', 'imgsz', 'date']
    datas = [args.loss, alpha, beta, zeta, args.attack, args.eps, args.model, args.meta_lr, args.sche, scheduler_args, args.task_num, args.imgsz, date]
    df = pd.DataFrame([datas], columns=columns)
    df.to_csv(f"./logs/train_log.csv", index=False, header=False, mode='a')

def check_args(args):
    # check argument
    if args.model not in ["conv3", "resnet18"]:
        print("select valid models")
        return False