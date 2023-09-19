#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
import  numpy as np

from    learner import Learner
from    copy import deepcopy
from    aRUBattack import aRUB
import advertorchMeta.attacks as attacks

from autoattackMeta import AutoAttack

# from loss import loss_function

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, device):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        #self.adv_lr = args.adv_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        # self.task_num = args.task_num
        # self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.device = device

        self.imgc = args.imgc
        self.imgsz = args.imgsz
        # self.eps = args.eps/255
        self.test_eps = args.test_eps/255
        self.iter = args.iter

        # self.loss = "no"
        # self.loss_function = loss_function(args.alpha, args.beta, args.zeta)
        
        self.args = args

        self.net = Learner(config, self.imgc, self.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        #self.meta_optim_adv = optim.Adam(self.net.parameters(), lr=self.adv_lr)

        # self.at = self.setAttack(args.attack, self.eps, self.iter)
        self.test_at = self.set_test_attack(args.test_attack, self.test_eps, self.iter)
        self.aa = args.auto_attack

    def setAttack(self, str_at, e, iter):
        if str_at == "PGD-L1":
            return attacks.L1PGDAttack(self.net, eps=e, nb_iter=iter) # 10., 40
        elif str_at == "PGD-L2":
            return attacks.L2PGDAttack(self.net, eps=e, nb_iter=iter) # 0.3, 40
        elif str_at == "PGD-Linf":
            return attacks.LinfPGDAttack(self.net, eps=e, nb_iter=iter) # 0.3, 40
        elif str_at == "FGSM":
            return attacks.GradientSignAttack(self.net, eps=e) # 0.3
        elif str_at == "BIM-L2":
            return attacks.L2BasicIterativeAttack(self.net, eps=e, nb_iter=iter) # 0.1, 10
        elif str_at == "BIM-Linf":
            return attacks.LinfBasicIterativeAttack(self.net, eps=e, nb_iter=iter) # 0.1, 10
        elif str_at == "MI-FGSM":
            return attacks.MomentumIterativeAttack(self.net, eps=e, nb_iter=iter) # 0.3, 40
        elif str_at == "CnW":
            return attacks.CarliniWagnerL2Attack(self.net, self.n_way, binary_search_steps=9, max_iterations=iter*10) # 9, 10000
        elif str_at == "EAD":
            return attacks.ElasticNetL1Attack(self.net, self.n_way, binary_search_steps=9, max_iterations=iter*10) # 9, 10000
        elif str_at == "DDN":
            return attacks.DDNL2Attack(self.net, nb_iter=iter*3) # 100
        elif str_at == "Single-pixel":
            return attacks.SinglePixelAttack(self.net, max_pixels=iter*3) # 100
        elif str_at == "DeepFool" or str_at == "Deepfool":
            return attacks.DeepfoolLinfAttack(self.net, self.n_way, nb_iter=iter*2, eps=e) # 50, 0.1
        elif str_at == "aRUB":
            return aRUB(self.net, rho=e, q=1, n_way=self.n_way, k_qry=self.k_qry, imgc=self.imgc, imgsz=self.imgsz, device=self.device)
        else:
            print("wrong type Attack")
            exit()

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        # step_acc = []

        querysz = x_qry.size(0)

        #corrects = [0 for _ in range(self.update_step_test + 1)]
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        # corrects_adv = [0 for _ in range(self.update_step_test + 1)]
        # corrects_adv_prior = [0 for _ in range(self.update_step_test + 1)]

        #corrects = 0
        #corrects_adv = 0

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # self.test_at = AutoAttack(net, norm=self.args.auto_norm, eps=self.test_eps, version=self.args.auto_version, device=self.device)
        # 1. run the i-th task and compute loss for k=0
        # acc, acc_adv = self.cal_acc(net, x_qry, y_qry, net.parameters())
        # step_acc.append([acc, acc_adv])

        # logits_q = net(x_spt, fast_weights, bn_training=True)
        # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        # #find the correct index
        # #corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
        # correct = torch.eq(pred_q, label).sum().item()  # convert to numpy
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # acc, acc_adv = self.cal_acc(net, x_qry, y_qry, fast_weights)
        # step_acc.append([acc, acc_adv])
        
        '''
        # Adversaruak Attack
        if need_adv:
            data = x_qry
            label = y_qry
            optimizer.zero_grad()
            adv_inp_adv = self.test_at.perturb(fast_weights, data, label)
        '''
        # loss_fn = self.loss_function.set_loss(False, "no")
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
        

            #loss, _, _, _, _ = loss_fn(net, fast_weights, x_spt, y_spt, _, 0)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            # loss_q = F.cross_entropy(logits_q, y_qry)
            
            # test step
            # acc, acc_adv = self.cal_acc(net, x_qry, y_qry, fast_weights)
            # step_acc.append([acc, acc_adv])


            if k==self.update_step_test - 1:
                optimizer.zero_grad()
                data = x_qry
                label = y_qry
                #loss_fn = self.loss_function.set_loss(True, self.loss)
                #loss_q, _, _, _ = loss_fn(net, fast_weights, data, label, self.test_at)
                with torch.no_grad():
                    logits_q = net(data, fast_weights, bn_training=True)
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    #find the correct index
                    #corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
                    correct = torch.eq(pred_q, label).sum().item()  # convert to numpy
                if not self.aa:
                    logits_q_adv = self.test_at.perturb(fast_weights, data, label)
                    with torch.no_grad():
                        pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                        correct_adv = torch.eq(pred_q_adv, label).sum().item()
        
                
                    
                # # Adversarial Attack
                # if need_adv and not self.aa:
                    
                    
                #     # correct_adv_prior = torch.eq(pred_q_adv[corr_ind], label[corr_ind]).sum().item()
                #     corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv
                #     # corrects_adv_prior[k + 1] = corrects_adv_prior[k + 1] + correct_adv_prior/len(corr_ind)
        if self.aa:
            accs_adv = self.test_at.run_standard_evaluation(fast_weights, x_qry, y_qry)
            # accs_adv_prior = 0

        del net

        accs = correct / querysz

        if not self.aa:
            accs_adv = correct_adv / querysz
            # accs_adv_prior = np.array(corrects_adv_prior)

        return accs, accs_adv #, step, accs_adv_prior
    
    
    def set_model(self, model):
        self.net.load_state_dict(model) # if model is only parameters

    def set_test_attack(self, attack, eps=6, iter=10, auto_list=[]):
        if attack=="Auto Attack":
            if self.args.auto_version=="custom":
                self.test_at = AutoAttack(self.net, norm=self.args.auto_norm, eps=eps/255, version=self.args.auto_version, attacks_to_run=auto_list, device=self.device)
            else:
                self.test_at = AutoAttack(self.net, norm=self.args.auto_norm, eps=eps/255, version=self.args.auto_version, device=self.device)
        else:
            self.test_at = self.setAttack(attack, eps/255, iter=iter)
    
def main():
    pass


if __name__ == '__main__':
    main()

