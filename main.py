import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from    metaTrain import Meta
from    utils import *


from torch.utils.tensorboard import SummaryWriter


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0] 
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main(args):
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

    str_path = f"./logs/runs_table/"
    sum_str_path = set_str_path(args)
    
    add_log(args)
    fix_seed()
    
    writer = SummaryWriter(str_path + sum_str_path, comment=args.attack)

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

            # if not args.trades:
            #     accs, accs_adv, loss_q, loss_q_adv = maml(x_spt, y_spt, x_qry, y_qry)
            # else:
            #     accs, accs_adv, loss_q, loss_q_clean, loss_q_adv = maml.forward_trades(x_spt, y_spt, x_qry, y_qry)

            accs, accs_adv, losses_item = maml(x_spt, y_spt, x_qry, y_qry)
            
            if step % 40 == 0:
                print('step:', tot_step,'/',args.epoch*4000)
                print('\ttraining acc:', accs)
                print('\ttraining acc_adv:', accs_adv)
                # print('\tloss:', losses_item[0], losses_item[1], losses_item[2])
                writer.add_scalar("acc/train", accs, tot_step)
                writer.add_scalar("acc_adv/train", accs_adv, tot_step)
                writer.add_scalar("loss/train", losses_item[0], tot_step)
                writer.add_scalar("loss_clean/train", losses_item[1], tot_step)
                writer.add_scalar("loss_adv/train", losses_item[2], tot_step)

    dir_path = f'./models/{args.imgsz}/'
    os.makedirs(dir_path, exist_ok=True)
    str_path = sum_str_path.split("/")
    torch.save(maml.get_model(), f"{dir_path}{args.loss}_{str_path[2]}.pth")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Meta-learning options
    argparser.add_argument('--model', type=str, help='model architecture', default="conv3")
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=56)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    
    # Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=100)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    #argparser.add_argument('--adv_lr', type=float, help='adv-level learning rate', default=0.0002)
    argparser.add_argument('--alpha', type=float, help='R-MAML adv lr rate', default=0.2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # Adversarial training options
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--eps', type=float, help='training attack eps', default=6) # 6/255
    # argparser.add_argument('--rho', type=float, help='aRUB-rho', default=6) # 6/255
    argparser.add_argument('--iter', type=int, help='number of iterations for iterative attack', default=10)
    
    # Loss function options
    argparser.add_argument('--loss', type=str, help='AT, AT-WAR, trades, trades-WAR, no', default="AT")
    argparser.add_argument('--beta', type=float, help='using for trades', default=1.0)
    argparser.add_argument('--zeta', type=float, help='WAR parameter', default=30)
    argparser.add_argument('--sche', type=str, help='learning rate scheduler for meta_lr', default='')
    argparser.add_argument('--sche_arg1', type=float, help='learning rate scheduler argument 1', default=-1)
    argparser.add_argument('--sche_arg2', type=float, help='learning rate scheduler argument 2', default=-1)

    args = argparser.parse_args()

    check_result = check_args(args)
    if not check_result:
        exit()

    main(args)