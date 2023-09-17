
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

def set_config(args):
    if args.model == "conv3":
        config = [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('adaptive_max_pool2d', [1, 1]),
            ('flatten', []),
            ('linear', [args.n_way, 32])
        ]
    elif args.model == "resnet9":
        config = [
            ('conv2d', [64, 3, 3, 3, 1, 1]),
            ('bn', [64]),
            ('relu', [True]),
            ('conv2d', [128, 64, 3, 3, 1, 1]),
            ('bn', [128]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('res_basic', [2, 128, 128, 3, 3, 1, 1]),
            ('conv2d', [256, 128, 3, 3, 1, 1]),
            ('bn', [256]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [512, 256, 3, 3, 1, 1]),
            ('bn', [512]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('res_basic', [2, 512, 512, 3, 3, 1, 1]),
            ('adaptive_avg_pool2d', [1, 1]),
            ('flatten', []),
            ('linear', [args.n_way, 512])
        ]
    return config

def set_str_path(args):
    if args.loss == ("AT"):
        sum_str_path = f"{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.alpha}"
    elif args.loss == ("trades"):
        sum_str_path = f"{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.beta}"
    elif args.loss.find("WAR") >= 0:
        sum_str_path = f"{args.loss}/{args.attack}_{args.eps}_{args.meta_lr}_{args.zeta}"
    else:
        sum_str_path = f"{args.loss}/{args.meta_lr}"

    return sum_str_path

def set_pretrained_model(args, config):
    print(f"use pre-trained model {args.pretrained}")
    params = torch.load("./pretrained_models/"+args.pretrained)
    if args.model == "conv3":
        w = torch.nn.Parameter(torch.ones(args.n_way, 32))
        torch.nn.init.kaiming_normal_(w)
        params['vars.12'] = w
        params['vars.13'] = torch.nn.Parameter(torch.zeros(args.n_way))
        return params
    if args.model == "resnet9":
        w = torch.nn.Parameter(torch.ones(args.n_way, 512))
        torch.nn.init.kaiming_normal_(w)
        params['vars.32'] = w
        params['vars.33'] = torch.nn.Parameter(torch.zeros(args.n_way))
        return params

# def add_idx(str_path, args):
#     # path_list = str_path.split("/")
#     # paths = glob.glob(f"./models/{path_list[0]}/*")
#     # model_paths = [(os.path.basename(i))[:-4] for i in paths]

#     # idx = 0
#     # for path in model_paths:
#     #     if path.startswith(path_list[1]+"_"+path_list[2]):
#     #         idx += 1

#     alpha, beta, zeta = args.alpha, args.beta, args.zeta
    

#     if args.loss != "AT":
#         alpha = "-"
#     if args.loss != "trades":
#         beta = "-"
#     if args.loss != "AT-WAR" and args.loss != "trades-WAR":
#         zeta = "-"

#     now = datetime.now()
#     date = now.strftime('%Y-%m-%d %H:%M:%S')
#     columns = ['loss', 'alpha', 'beta', 'zeta', 'attack', 'eps', 'model', 'meta_lr', 'scheduler', 'scheduler_args', 'task_num', 'imgsz', 'date']
#     datas = [args.loss, alpha, beta, zeta, args.attack, args.eps, args.model, args.meta_lr, args.scheduler, scheduler_args, args.task_num, args.imgsz, date]
#     df = pd.DataFrame([datas], columns=columns)
#     # df.to_csv(f"./logs/train_log.csv", index=True, header=True, mode='w') # to make new csv file
#     df.to_csv(f"./logs/train_log.csv", index=True, header=False, mode='a')
#     #return idx

# def add_log(args):
#     alpha, beta, zeta = args.alpha, args.beta, args.zeta
    
#     if args.loss != "AT":
#         alpha = "-"
#     if args.loss != "trades":
#         beta = "-"
#     if args.loss != "AT-WAR" and args.loss != "trades-WAR":
#         zeta = "-"

#     scheduler_args = ""
#     if args.sche == "lambda":
#         scheduler_args = f"lambda : {args.sche_arg1}"
#     elif args.sche == "step":
#         scheduler_args = f"step: {args.sche_arg1} gamma: {args.sche_arg2}"
#     elif args.sche == "cosine":
#         scheduler_args = f"T max: {args.sche_arg1} eta-min: {args.sche_arg2}"
        
#     now = datetime.now()
#     date = now.strftime('%Y-%m-%d %H:%M:%S')
#     columns = ['loss', 'alpha', 'beta', 'zeta', 'attack', 'eps', 'model', 'meta_lr', 'scheduler', 'scheduler_args', 'task_num', 'imgsz', 'date']
#     datas = [args.loss, alpha, beta, zeta, args.attack, args.eps, args.model, args.meta_lr, args.sche, scheduler_args, args.task_num, args.imgsz, date]
#     df = pd.DataFrame([datas], columns=columns)
#     df.to_csv(f"./logs/train_log.csv", index=False, header=False, mode='a')


def check_args(args):
    # check argument
    if args.model not in ["conv3", "resnet18"]:
        print("select valid models")
        return False
    if args.device_num not in [0,1,2,3]: 
        print("GPU number can be 0, 1, 2, 3")
        return False
    if args.sche != "":
        if args.sche_arg1 == -1:
            print(f"learning rate scheduler argument Error")
            return False
        if args.sche == "AT-WAR" or args.sche == "trades-WAR":
            print()

