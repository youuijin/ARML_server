import  torch, os
import  numpy as np
import  scipy.stats
import  argparse
from    utils import *
from    learner import Learner

import torchvision 
import torchvision.transforms as transforms

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

    config = set_config(args)

    device = torch.device('cuda:'+str(args.device_num))

    str_path = f"./logs/train_pretrained/{args.model}/{args.lr}"

    fix_seed()
    
    writer = SummaryWriter(str_path)

    net = Learner(config, args.imgc, args.imgsz).to(device)

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
    dir_path = f'./pretrained_models/'
    os.makedirs(dir_path, exist_ok=True)
    torch.save(net.state_dict(), f"{dir_path}{args.model}_{args.lr}.pth")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Meta-learning options
    argparser.add_argument('--model', type=str, help='model architecture, conv3, resnet9', default="conv3")
    argparser.add_argument('--n_way', type=int, help='n way', default=5)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=56)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    
    # Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1000)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    args = argparser.parse_args()


    main(args)