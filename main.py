import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from    metaTrain import Meta

import  threading
import  queue
import  random
import pandas as pd
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

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
    #maml = Meta(args, config, device).to(device)

    model_paths =['aRUB_8.0_0.001_0.0001.pth']
    
    thread_list = []
    
    if args.test:
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
        df.to_csv(f"./logs/AAresult_csv/{args.auto_version}_{args.auto_norm}_{args.test_eps}_{t}.csv")

        exit()
    if args.attack=="aRUB": 
        bound = args.rho
    else:
        bound = args.eps
    if args.trades:
        sum_str_path = f"./logs/runs_table/trades/{args.imgsz}/{args.attack}/{bound}/{args.meta_lr}_{args.beta}"
    else:
        sum_str_path = f"./logs/runs_table/{args.imgsz}/{args.attack}/{bound}/{args.meta_lr}_{args.adv_lr}"
    writer = SummaryWriter(sum_str_path, comment=args.attack)
    print(sum_str_path)

    maml = Meta(args, config, device).to(device)
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    # batchsz here means total episode number
    mini = MiniImagenet('../', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry, batchsz=4000, resize=args.imgsz) # batch size = 4000 for small scale 
    
    tot_step = -args.task_num
    for _ in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            
            tot_step = tot_step + args.task_num

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            if not args.trades:
                accs, accs_adv, loss_q, loss_q_adv = maml(x_spt, y_spt, x_qry, y_qry)
            else:
                accs, accs_adv, loss_q, loss_q_clean, loss_q_adv = maml.forward_trades(x_spt, y_spt, x_qry, y_qry)
            
            if step % 10 == 0:
                print('step:', tot_step,'/',args.epoch*4000)
                print('\ttraining acc:', accs)
                print('\ttraining acc_adv:', accs_adv)
                writer.add_scalar("acc/train", accs, tot_step)
                writer.add_scalar("acc_adv/train", accs_adv, tot_step)
                writer.add_scalar("loss/train", loss_q.item(), tot_step)
                writer.add_scalar("loss_adv/train", loss_q_adv, tot_step)
                if args.trades:
                    writer.add_scalar("loss_clean/train", loss_q_clean, tot_step)

    
    
    if args.trades:
        str_path = f"trades/{args.imgsz}/{args.attack}_{bound}_{args.meta_lr}_{args.beta}"
    else: 
        str_path = f"{args.imgsz}/{args.attack}_{bound}_{args.meta_lr}_{args.adv_lr}"
    #str_path = str(args.imgsz)+"/noAttack_"+str(args.meta_lr)
    torch.save(maml.get_model(), './models/'+str_path+".pth")

def test_model(maml, path, device):
    if path=="":
        print("Enter test model path")
        exit()
    str_path = './models/'+str(args.imgsz)+"/"+path
    if args.trades:
        str_path = './models/trades/'+str(args.imgsz)+"/"+path
    model = torch.load(str_path)
    maml.set_model(model)

    mini_test = MiniImagenet('../', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry, batchsz=50, resize=args.imgsz) # batch size = 40 for small scale
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
        accsadvpr_all_test = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                            x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, accs_adv, accs_adv_prior = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)
            accsadv_all_test.append(accs_adv)
            accsadvpr_all_test.append(accs_adv_prior)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        accs_adv = np.array(accsadv_all_test).mean(axis=0).astype(np.float16)
        accs_adv_prior = np.array(accsadvpr_all_test).mean(axis=0).astype(np.float16)
        if args.auto_attack:
            q.put([path, accs[-1], accs_adv])
        else:
            q.put([path, attack_name, args.test_eps, accs[-1], accs_adv[-1]])



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
    argparser.add_argument('--epoch', type=int, help='epoch number', default=25)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=40)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--adv_lr', type=float, help='adv-level learning rate', default=0.0002)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # Adversarial training options
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--eps', type=float, help='training attack eps', default=6) # 6/255
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=6) # 6/255
    argparser.add_argument('--iter', type=int, help='number of iterations for iterative attack', default=10)
    argparser.add_argument('--trades', action='store_true', help='using trades adversarial training', default=False)
    argparser.add_argument('--beta', type=float, default=1.0)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='testing atttack eps', default=6) # 6/255
    
    # to test models
    argparser.add_argument('--test', action='store_true', default=False)
    argparser.add_argument('--path', type=str, help='test model path', default="")
    argparser.add_argument('--auto_attack', action='store_true', default=False)
    argparser.add_argument('--auto_version', type=str, default="standard")
    

    args = argparser.parse_args()

    main(args)