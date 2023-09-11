import  torch, os
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from    utils import *
from    learner import Learner

import torchvision 
import torchvision.transforms as transforms
from torchvision import models

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

    str_path = f"./logs/train_pretrained/resnet18/{args.lr}_StepLR_save"
    # sum_str_path = set_str_path(args)
    
    # add_log(args)
    fix_seed()
    
    writer = SummaryWriter(str_path)

    # net = Learner(config, args.imgc, args.imgsz).to(device)
    net = models.resnet18()
    input_features_fc_layer = net.fc.in_features # fc layer 채널 수 얻기
    net.fc = torch.nn.Linear(input_features_fc_layer, args.n_way, bias=False) # fc layer 수정'
    net = net.to(device)

    # tmp = filter(lambda x: x.requires_grad, maml.parameters())
    # num = sum(map(lambda x: np.prod(x.shape), tmp))

    # batchsz here means total episode number
    # mini = MiniImagenet('../', mode='train', n_way=args.n_way, k_shot=args.k_spt,
    #                     k_query=args.k_qry, batchsz=4000, resize=args.imgsz) # batch size = 4000 for small scale 
    transform = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0,0,0), (1,1,1))])
    
    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.task_num, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.task_num, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,40], gamma=0.5)
    for epoch in range(args.epoch):
        net.train()
        train_loss = 0
        train_acc = 0
        for i, data in enumerate(trainloader):
            x, y = data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            pred = torch.nn.functional.softmax(output, dim=1).argmax(dim=1)
            correct = torch.eq(pred, y).sum().item()  # convert to numpy
            train_acc += correct

            train_loss += loss.item()
        scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        for i, data in enumerate(testloader):
            x, y = data
            x, y = x.to(device), y.to(device)

            output = net(x)
            loss = criterion(output, y)
            loss.backward()

            pred = torch.nn.functional.softmax(output, dim=1).argmax(dim=1)
            correct = torch.eq(pred, y).sum().item()  # convert to numpy
            test_acc += correct
            
            test_loss += loss.item()

        train_loss/=len(trainset)
        train_acc/=len(trainset)
        test_loss/=len(testset)
        test_acc/=len(testset)
            
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", test_acc, epoch)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", test_loss, epoch)

        print(f"epoch: {epoch}\tlr: {scheduler.get_last_lr()}\ttrain acc: {round(train_acc*100, 2)}%, test acc: {round(test_acc*100, 2)}%")
        #print(f"epoch: {epoch}\ttrain acc: {round(train_acc*100, 2)}%, test acc: {round(test_acc*100, 2)}%")
    dir_path = f'./pretrained_models/'
    os.makedirs(dir_path, exist_ok=True)
    torch.save(net.state_dict(), f"{dir_path}resnet18_{args.lr}_StepLR.pth")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Meta-learning options
    # argparser.add_argument('--model', type=str, help='model architecture', default="conv3")
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    # argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    # argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=56)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    
    # Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1000)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    #argparser.add_argument('--adv_lr', type=float, help='adv-level learning rate', default=0.0002)
    # argparser.add_argument('--alpha', type=float, help='R-MAML adv lr rate', default=0.2)
    # argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    # argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    # argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # Adversarial training options
    # argparser.add_argument('--attack', type=str, default="aRUB")
    # argparser.add_argument('--eps', type=float, help='training attack eps', default=6) # 6/255
    # argparser.add_argument('--rho', type=float, help='aRUB-rho', default=6) # 6/255
    # argparser.add_argument('--iter', type=int, help='number of iterations for iterative attack', default=10)
    
    # Loss function options
    # argparser.add_argument('--loss', type=str, help='AT, AT-WAR, trades, trades-WAR, no', default="AT")
    # argparser.add_argument('--beta', type=float, help='using for trades', default=1.0)
    # argparser.add_argument('--zeta', type=float, help='WAR parameter', default=30)
    # argparser.add_argument('--sche', type=str, help='learning rate scheduler for meta_lr', default='')
    # argparser.add_argument('--sche_arg1', type=float, help='learning rate scheduler argument 1', default=-1)
    # argparser.add_argument('--sche_arg2', type=float, help='learning rate scheduler argument 2', default=-1)

    args = argparser.parse_args()

    # check_result = check_args(args)
    # if not check_result:
    #     exit()

    main(args)