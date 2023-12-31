import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from    metaTrain import Meta
from    utils import *
from    tqdm.auto import tqdm


from torch.utils.tensorboard import SummaryWriter


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0] 
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main(args):
    
    # """"""
    # params = torch.load("./pretrained_models/"+args.pretrained)
    # # print(params)
    # # print(type(params))
    # for idx, data in params.items():
    #     print(idx, data.shape)

    # exit()
    # """"""

    fix_seed()
    config = set_config(args)
    device = torch.device('cuda:'+str(args.device_num))

    if args.pretrained == "":
        str_path = f"./logs/runs_table/{args.model}"
    else:
        str_path = f"./logs/runs_table_pre/{args.model}/{args.pretrained}/"
    sum_str_path = set_str_path(args)
    print(str_path + sum_str_path)

    writer = SummaryWriter(str_path + sum_str_path, comment=args.attack)

    maml = Meta(args, config, device).to(device)

    if args.pretrained != "":
        params = set_pretrained_model(args, config)
        maml.set_model(params)
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    # batchsz here means total episode number
    mini = MiniImagenet('../', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry, batchsz=args.train_tasks, resize=args.imgsz) # batch size = 4000 for small scale 
    mini_test = MiniImagenet('../', mode='val', n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry, batchsz=args.val_tasks, resize=args.imgsz)
    
    for epoch in tqdm(range(args.epoch), position=0, desc='epoch'):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        
        accses = 0
        accses_adv = 0
        losses_1 = 0
        losses_2 = 0
        losses_3 = 0
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm(db, position=1, desc='step', leave=False)):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs, accs_adv, losses_item = maml(x_spt, y_spt, x_qry, y_qry)
            accses += accs*x_spt.shape[0]
            accses_adv += accs_adv*x_spt.shape[0]
            losses_1 += losses_item[0]*x_spt.shape[0]
            losses_2 += losses_item[1]*x_spt.shape[0]
            losses_3 += losses_item[2]*x_spt.shape[0]
            
            # if step % 10 == 0:
            #     print('epoch:', epoch, '/', args.epoch, '\tstep:', step*args.task_num, '/', args.train_tasks, '\tacc:', accs, '\tacc_adv:', accs_adv)
        
        # print('\ttraining acc:', accses/args.train_tasks)
        # print('\ttraining acc_adv:', accses_adv/args.train_tasks)
        writer.add_scalar("acc/train", accses/args.train_tasks, epoch)
        writer.add_scalar("acc_adv/train", accses_adv/args.train_tasks, epoch)
        writer.add_scalar("loss/train", losses_1/args.train_tasks, epoch)
        writer.add_scalar("loss_clean/train", losses_2/args.train_tasks, epoch)
        writer.add_scalar("loss_adv/train", losses_3/args.train_tasks, epoch)
        if args.loss.find("WAR")>=0:
            writer.add_scalar("lambda", maml.get_lambda(), epoch)

        # Validation 
        if epoch % 2 == 0:
            db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            
            accs_all = 0
            accs_adv_all = 0
            losses = 0

            for _, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                            x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                accs, accs_adv, loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                accs_all += accs
                accs_adv_all += accs_adv
                losses += loss
                
            writer.add_scalar("acc/val", accs_all/args.val_tasks, epoch)
            writer.add_scalar("acc_adv/val", accs_adv_all/args.val_tasks, epoch)
            writer.add_scalar("loss/val", losses/args.val_tasks, epoch)

    dir_path = f'./models/{args.model}/'
    os.makedirs(dir_path, exist_ok=True)
    str_path = sum_str_path.split("/")
    torch.save(maml.get_model(), f"{dir_path}{args.loss}_{str_path[1]}.pth")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Meta-learning options
    argparser.add_argument('--model', type=str, help='model architecture, conv3, resnet9', default="conv3")
    argparser.add_argument('--pretrained', type=str, help='pretrained model path', default="")
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=56)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--train_tasks', type=int, help='train task num', default=4000)
    argparser.add_argument('--val_tasks', type=int, help='validation task num', default=15)
    
    # Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=100)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--alpha', type=float, help='R-MAML adv lr rate', default=0.2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # Adversarial training options
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--eps', type=float, help='training attack eps', default=6) # 6/255
    argparser.add_argument('--iter', type=int, help='number of iterations for iterative attack', default=10)
    
    # Loss function options
    argparser.add_argument('--loss', type=str, help='AT, AT-WAR, trades, trades-WAR, no', default="AT")
    argparser.add_argument('--beta', type=float, help='using for trades', default=1.0)
    argparser.add_argument('--zeta', type=float, help='WAR parameter', default=1.0)

    args = argparser.parse_args()

    main(args)