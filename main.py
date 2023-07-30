import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from metaTrain import Meta
from torchvision import models, transforms
import random

from torch.utils.tensorboard import SummaryWriter

import pandas as pd

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0] 
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main(args):
    seed = 222
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    maml = Meta(args, config, device).to(device)

    if args.test:
        test_model(maml, args.path, device)
        exit()
    if args.attack=="aRUB": 
        bound = args.rho
    else:
        bound = args.eps
    sum_str_path = "./logs/runs_table/"+str(args.imgsz)+"/"+args.attack+"/"+str(bound)+"/"+str(args.meta_lr)+"_"+str(args.adv_lr)
    writer = SummaryWriter(sum_str_path, comment=args.attack+"/"+args.test_attack)
    print(sum_str_path)
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    # batchsz here means total episode number
    mini = MiniImagenet('../', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry, batchsz=1000, resize=args.imgsz) # batch size = 4000 for small scale 
    
    # mini_val = MiniImagenet('../', mode='val', n_way=args.n_way, k_shot=args.k_spt,
    #                          k_query=args.k_qry,
    #                          batchsz=100, resize=args.imgsz)

    tot_step = -args.task_num
    for _ in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            tot_step = tot_step + args.task_num

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            # logits_a = maml.print_logits(x_qry[0])
            # model = torch.load('./models/'+str(args.imgsz)+"/"+"aRUB_2_0.001_0.0002.pth")
            # print(type(model))
            # maml.set_model(model)
            # logits_b = maml.print_logits(x_qry[0])
            # print(torch.equal(logits_a, logits_b))
            # exit()
            accs, accs_adv, loss_q, loss_q_adv = maml(x_spt, y_spt, x_qry, y_qry)
            
            
            if step % 2 == 0:
                print('step:', tot_step,'/',args.epoch*1000)
                print('\ttraining acc:', accs)
                print('\ttraining acc_adv:', accs_adv)
                writer.add_scalar("acc/train", accs[-1], tot_step)
                writer.add_scalar("acc_adv/train", accs_adv[-1], tot_step)
                writer.add_scalar("loss/train", loss_q, tot_step)
                writer.add_scalar("loss_adv/train", loss_q_adv, tot_step)
        ''' no validation 
        if epoch%10 == 0:
            attack_list = ["BIM_L2", "BIM_Linf", "CnW", "DDN", "EAD", "FGSM", "MI_FGSM", "PGD_L1", "PGD_L2", "PGD_Linf", "Single_pixel", "DeepFool"]
            db_val = DataLoader(mini_val, 1, shuffle=True, num_workers=0, pin_memory=True)
            for _, attack_name in enumerate(attack_list):
                attack_writer = SummaryWriter('./val_acc/'+str(args.imgsz)+"/"+args.attack+"/"+str(bound)+"/"+str(args.meta_lr)+"_"+str(args.adv_lr)+"/"+attack_name, comment=str(args.imgsz)+"/"+args.attack+"/"+str(bound)+"/"+str(args.meta_lr)+"_"+str(args.adv_lr))
            
                maml.set_test_attack(attack_name)
                
                accs_all_test = []
                accsadv_all_test = []
                accsadvpr_all_test = []
                loss_all_test = []
                loss_adv_all_test = []
                
                for x_spt, y_spt, x_qry, y_qry in db_val:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                    x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs, accs_adv, accs_adv_prior, loss_q, loss_q_adv = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)
                    accsadv_all_test.append(accs_adv)
                    accsadvpr_all_test.append(accs_adv_prior)
                    loss_all_test.append(loss_q.item())
                    loss_adv_all_test.append(loss_q_adv.item())

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                accs_adv = np.array(accsadv_all_test).mean(axis=0).astype(np.float16)
                accs_adv_prior = np.array(accsadvpr_all_test).mean(axis=0).astype(np.float16)
                loss_q = np.array(loss_all_test).mean()
                loss_q_adv = np.array(loss_adv_all_test).mean()
                print(attack_name)
                print('Val acc:', accs)
                print('Val acc_adv:', accs_adv)
                print('Val acc_adv_prior:', accs_adv_prior)
                
                attack_writer.add_scalar("acc/val_epoch", accs[-1],epoch)
                attack_writer.add_scalar("acc_adv/val_epoch", accs_adv[-1],epoch)
                attack_writer.add_scalar("loss/epoch", loss_q, epoch)
                attack_writer.add_scalar("loss_adv/epoch", loss_q_adv, epoch)
        '''
        
    str_path = str(args.imgsz)+"/"+args.attack+"_"+str(bound)+"_"+str(args.meta_lr)+"_"+str(args.adv_lr)
    torch.save(maml.get_model(), './models/'+str_path+".pth")

def test_model(maml, path, device):
    if path=="":
        print("Enter test model path")
        exit()
    model = torch.load('./models/'+str(args.imgsz)+"/"+path)
    maml.set_model(model)

    mini_test = MiniImagenet('../', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry, batchsz=40, resize=args.imgsz) # batch size = 40 for small scale
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
    attack_list = ["BIM_L2", "BIM_Linf", "CnW", "DDN", "EAD", "FGSM", "MI_FGSM", "PGD_L1", "PGD_L2", "PGD_Linf", "Single_pixel", "DeepFool"]
    for _, attack_name in enumerate(attack_list):
        writer = SummaryWriter('./logs/test_acc/'+path+"/"+attack_name+"/"+str(args.test_eps), comment=path)
        tot_writer = SummaryWriter('./logs/test_acc/'+attack_name+"/"+str(args.test_eps)+"/"+path, comment=path)
        
        maml.set_test_attack(attack_name, eps=args.test_eps)
        accs_all_test = []
        accsadv_all_test = []
        accsadvpr_all_test = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                            x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, accs_adv, accs_adv_prior, _, _ = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)
            accsadv_all_test.append(accs_adv)
            accsadvpr_all_test.append(accs_adv_prior)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        accs_adv = np.array(accsadv_all_test).mean(axis=0).astype(np.float16)
        accs_adv_prior = np.array(accsadvpr_all_test).mean(axis=0).astype(np.float16)

        #-4.3947e-01

        print(attack_name)
        print('Test acc:', accs[-1])
        print('Test acc_adv:', accs_adv[-1])
        print('Test acc_adv_prior:', accs_adv_prior[-1])
        
        writer.add_scalar("accs", round(accs_adv[-1]*10000), round(accs[-1]*10000)) # 가로축 SA*1000, 세로축 RA*1000
        writer.add_scalar("accs/"+path, round(accs_adv[-1]*10000), round(accs[-1]*10000)) # 가로축 SA*1000, 세로축 RA*1000
        tot_writer.add_scalar("acc/"+attack_name, round(accs_adv[-1]*10000), round(accs[-1]*10000))


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
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--adv_lr', type=float, help='adv-level learning rate', default=0.0002)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # adversarial attack options
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--eps', type=float, help='training attack eps', default=2) # 2/255
    argparser.add_argument('--test_eps', type=float, help='testing atttack eps', default=2) # 2/255
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=2) # 2/255

    # to test models
    argparser.add_argument('--test', action='store_true', default=False)
    argparser.add_argument('--path', type=str, help='test model path', default="")

    args = argparser.parse_args()

    main(args)