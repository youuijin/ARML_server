import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from    metaTest import Meta

from torch.utils.tensorboard import SummaryWriter

import  threading
import  queue
import  random
import pandas as pd
from datetime import datetime

import glob
import os

from utils import *

q = queue.Queue()

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0] 
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main(args):
    fix_seed()

    config = set_config(args)
    device = torch.device('cuda:'+str(args.device_num))

    paths = glob.glob(args.dir_path+"*")
    #model_paths = [os.path.basename(i) for i in paths]
    model_paths = ['AT_PGD-Linf_6_0.001_0.0008.pth']
    thread_list = []

    if args.auto_version == "custom":
        save_str = f"{args.auto_version}_{args.auto_custom}_{args.auto_norm}_{args.test_eps}.csv"
    else:
        save_str = f"{args.auto_version}_{args.auto_norm}_{args.test_eps}.csv"
    
    if os.path.isfile(f"./logs/AAresult_csv/{args.model}/{save_str}") == False:
        df = pd.DataFrame([], columns=["model", "SA", "RA", "date"])
        df.to_csv(f"./logs/AAresult_csv/{save_str}", header=True, mode='w')
    
    csvs = pd.read_csv(f"./logs/AAresult_csv/{save_str}")
    
    csv_models = []
    for _, data in csvs.iterrows():
        csv_models.append(data['model'])
    
    for model in model_paths:
        if model in csv_models:
            continue
        maml = Meta(args, config, device).to(device)
        t = threading.Thread(target=test_model, args=(maml, model, device, ))#TODO
        t.daemon = True
        t.start()
        thread_list.append(t)
    
    for thread in thread_list:
        print(thread)
        thread.join()
    qsize = q.qsize()
    datas = []

    for _ in range(qsize):
        temp = q.get()
        datas.append(temp)

    if args.auto_attack:
        #datas = sorted(datas, key=lambda x: (x[0]))
        df = pd.DataFrame(datas, columns=["model", "SA", "RA", "date"])
    else:
        datas = sorted(datas, key=lambda x: (x[0], x[1]))
        df = pd.DataFrame(datas, columns=["model", "attack", "eps", "SA", "RA"])
    
    df.to_csv(f"./logs/AAresult_csv/{save_str}", header=False, mode='a')

def test_model(maml, path, device):
    if path=="":
        print("Enter test model path")
        exit()
    str_path = args.dir_path + path

    model = torch.load(str_path)
    maml.set_model(model)
    
    mini_test = MiniImagenet('../', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry, batchsz=10, resize=args.imgsz) # batch size = 50 for small scale
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
    auto_list = []
    if args.auto_attack:
        attack_list = ["Auto Attack"]
        if args.auto_version == 'custom':
            auto_attacks = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            for i in range(4):
                if args.auto_custom[i]=='1':
                    auto_list.append(auto_attacks[i])
            if len(auto_list)==0:
                print("auto-custom mode must has at least 1 attack")
                exit()
    else:
        attack_list = ["BIM_L2", "BIM_Linf", "CnW", "DDN", "EAD", "FGSM", "MI_FGSM", "PGD_L1", "PGD_L2", "PGD_Linf", "Single_pixel", "DeepFool"]
    for _, attack_name in enumerate(attack_list):
        fix_seed()
        #writer = SummaryWriter("./steps/qry/"+path, comment=attack_name)
        print(path, attack_name)
        
        maml.set_test_attack(attack_name, eps=args.test_eps, iter=args.iter, auto_list = auto_list)
        accs_all_test = []
        accsadv_all_test = []

        # step_accs = [0 for i in range(args.update_step_test+1)]
        # step_accs_adv = [0 for i in range(args.update_step_test+1)]

        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                            x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, accs_adv = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)
            accsadv_all_test.append(accs_adv)

            # step_accs = [step_accs[i]+step_acc[i][0] for i in range(args.update_step_test+1)]
            # step_accs_adv = [step_accs_adv[i]+step_acc[i][1] for i in range(args.update_step_test+1)]

        # for i in range(args.update_step_test+1):
        #     writer.add_scalar("acc", step_accs[i]/test_num, i)
        #     writer.add_scalar("acc_adv", step_accs_adv[i]/test_num, i)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        accs_adv = np.array(accsadv_all_test).mean(axis=0).astype(np.float16)

        t = datetime.today().strftime("%m%d%H%M%S")
        if args.auto_attack:
            q.put([path, accs, accs_adv, t])
        else:
            q.put([path, attack_name, args.test_eps, accs, accs_adv, t])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Meta-learning options
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=56)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    
    # Training options
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # Adversarial training options

    argparser.add_argument('--iter', type=int, help='number of iterations for iterative attack', default=10)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD-Linf")
    argparser.add_argument('--test_eps', type=float, help='testing atttack eps', default=6) # 6/255
    
    # to test models
    argparser.add_argument('--dir_path', type=str, help='test model directory path', default="./models/56/")
    argparser.add_argument('--auto_attack', action='store_true', default=True)
    argparser.add_argument('--auto_version', help='standard, plus, custom', type=str, default="standard")
    argparser.add_argument('--auto_custom', type=str, help='apgd-ce, apgd-t, fab-t, sqaure', default="1001")
    
    argparser.add_argument('--auto_norm', help='L1, L2, Linf', type=str, default="Linf")
    argparser.add_argument('--alpha', type=float, help='hyper-parameter for R-MAML-AT', default=0.2)
    argparser.add_argument('--beta', type=float, help='hyper-parameter for R-MAML-AT', default=1.0)
    argparser.add_argument('--zeta', type=float, help='hyper-parameter for R-MAML-AT', default=10)
    argparser.add_argument('--model', type=str, help='resnet9, conv3', default="conv3")

    args = argparser.parse_args()

    main(args)