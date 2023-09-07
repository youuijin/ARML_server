import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from    metaTest import Meta

import  threading
import  queue
import  random
import pandas as pd
from datetime import datetime

import glob
import os

q = queue.Queue()

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0] 
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main(args):
    fix_seed()

    s = (args.imgsz-2)//2
    s = (s-2)//2
    s = s-3

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
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * s * s])
    ]

    device = torch.device('cuda:'+str(args.device_num))


    paths = glob.glob(args.dir_path+"*")
    model_paths = [os.path.basename(i) for i in paths]
    #print(model_paths)
    #print(len(model_paths))
    # model_paths = ['aRUB_6_0.001_2.0.pth', 'aRUB_6_0.01_10.0.pth', 'PGD_Linf_6_0.001_15.0.pth', 'PGD_Linf_6_0.001_0.7.pth', 'PGD_Linf_6_0.001_5.0.pth', 'PGD_Linf_6_0.001_10.0.pth', 'aRUB_6_0.005_10.0.pth', 'aRUB_6_0.001_15.0.pth', 'DeepFool_6_0.001_5.0.pth', 'aRUB_6_0.0005_10.0.pth', 'aRUB_6_0.005_5.0.pth', 'PGD_L2_6_0.001_1.0.pth', 'PGD_L2_6_0.001_5.0.pth', 'DeepFool_6_0.001_1.0.pth', 'aRUB_6_0.001_5.0.pth', 'PGD_Linf_6_0.001_0.5.pth', 'aRUB_6_0.001_1.0.pth', 'PGD_Linf_6_0.001_1.0.pth', 'aRUB_6_0.001_10.0.pth', 'PGD_Linf_6_0.001_2.0.pth', 'PGD_L2_6_0.001_10.0.pth']    
    thread_list = []
    
    for model in model_paths:
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
        datas = sorted(datas, key=lambda x: (x[0]))
        df = pd.DataFrame(datas, columns=["model", "SA", "RA"])
    else:
        datas = sorted(datas, key=lambda x: (x[0], x[1]))
        df = pd.DataFrame(datas, columns=["model", "attack", "eps", "SA", "RA"])
    t = datetime.today().strftime("%m%d%H%M%S")

    if args.auto_no_at:
        df.to_csv(f"./logs/AAresult_csv/at_{args.auto_version}_{args.auto_norm}_{args.test_eps}_{t}.csv")
    else:
        df.to_csv(f"./logs/AAresult_csv/nat_{args.auto_version}_{args.auto_norm}_{args.test_eps}_{t}.csv")

def test_model(maml, path, device):
    if path=="":
        print("Enter test model path")
        exit()
    str_path = args.dir_path + path

    model = torch.load(str_path)
    maml.set_model(model)

    loss_type = path.split('_')[0]
    if loss_type!="no":
        loss_arg = path.split('_')[4]
        maml.set_loss(loss_type, loss_arg)
        maml.set_attack(path.split('_')[1], float(path.split('_')[2]))
    
    

    mini_test = MiniImagenet('../', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry, batchsz=50, resize=args.imgsz) # batch size = 50 for small scale
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
    if args.auto_attack:
        attack_list = ["Auto Attack"]
    else:
        attack_list = ["BIM_L2", "BIM_Linf", "CnW", "DDN", "EAD", "FGSM", "MI_FGSM", "PGD_L1", "PGD_L2", "PGD_Linf", "Single_pixel", "DeepFool"]
    for _, attack_name in enumerate(attack_list):
        fix_seed()
        print(path, attack_name)
        
        maml.set_test_attack(attack_name, eps=args.test_eps, iter=args.iter)
        accs_all_test = []
        accsadv_all_test = []
        #accsadvpr_all_test = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                            x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, accs_adv = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)
            accsadv_all_test.append(accs_adv)
            #accsadvpr_all_test.append(accs_adv_prior)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        accs_adv = np.array(accsadv_all_test).mean(axis=0).astype(np.float16)
        #accs_adv_prior = np.array(accsadvpr_all_test).mean(axis=0).astype(np.float16)
        if args.auto_attack:
            q.put([path, accs, accs_adv])
        else:
            q.put([path, attack_name, args.test_eps, accs, accs_adv])

def fix_seed():
    seed = 222
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    # argparser.add_argument('--epoch', type=int, help='epoch number', default=25)
    # argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=40)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--adv_lr', type=float, help='adv-level learning rate', default=0.0002)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # Adversarial training options
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--eps', type=float, help='training attack eps', default=6) # 6/255
    # argparser.add_argument('--rho', type=float, help='aRUB-rho', default=6) # 6/255
    argparser.add_argument('--iter', type=int, help='number of iterations for iterative attack', default=10)
    # argparser.add_argument('--trades', action='store_true', help='using trades adversarial training', default=False)
    argparser.add_argument('--loss', type=str, help='R-MAML, R-MAML-trades, trades, WAR, no', default="R-MAML")

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD-Linf")
    argparser.add_argument('--test_eps', type=float, help='testing atttack eps', default=6) # 6/255
    
    # to test models
    argparser.add_argument('--dir_path', type=str, help='test model directory path', default="./models/56/")
    argparser.add_argument('--auto_attack', action='store_true', default=False)
    argparser.add_argument('--auto_version', help='standard, plus', type=str, default="standard")
    argparser.add_argument('--auto_norm', help='L1, L2, Linf', type=str, default="Linf")
    argparser.add_argument('--auto_no_at', action='store_true', help='if set true, no adversarial training at meta test phase', default=False)
    argparser.add_argument('--alpha', type=float, help='hyper-parameter for R-MAML-AT', default=0.2)
    argparser.add_argument('--beta', type=float, help='hyper-parameter for R-MAML-AT', default=1.0)
    argparser.add_argument('--zeta', type=float, help='hyper-parameter for R-MAML-AT', default=10)

    args = argparser.parse_args()

    main(args)