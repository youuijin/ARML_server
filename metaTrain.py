#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy
from aRUBattack import aRUB
import advertorchMeta.attacks as attacks

# from utils import setAttack

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
        self.adv_lr = args.adv_lr
        self.rho = args.rho
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.device = device

        self.imgc = args.imgc
        self.imgsz = args.imgsz
        self.eps = args.eps/255
        self.test_eps = args.test_eps/255
        self.iter = args.iter
        
        self.args = args

        self.net = Learner(config, self.imgc, self.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_optim_adv = optim.Adam(self.net.parameters(), lr=self.adv_lr)

        self.at = self.setAttack(args.attack, self.eps, self.iter)
        self.test_at = self.setAttack(args.test_attack, self.test_eps, self.iter)

    def setAttack(self, str_at, e, iter):
        if str_at == "PGD_L1":
            return attacks.L1PGDAttack(self.net, eps=e, nb_iter=iter) # 10., 40
        elif str_at == "PGD_L2":
            return attacks.L2PGDAttack(self.net, eps=e, nb_iter=iter) # 0.3, 40
        elif str_at == "PGD_Linf":
            return attacks.LinfPGDAttack(self.net, eps=e, nb_iter=iter) # 0.3, 40
        elif str_at == "FGSM":
            return attacks.GradientSignAttack(self.net, eps=e) # 0.3
        elif str_at == "BIM_L2":
            return attacks.L2BasicIterativeAttack(self.net, eps=e, nb_iter=iter) # 0.1, 10
        elif str_at == "BIM_Linf":
            return attacks.LinfBasicIterativeAttack(self.net, eps=e, nb_iter=iter) # 0.1, 10
        elif str_at == "MI_FGSM":
            return attacks.MomentumIterativeAttack(self.net, eps=e, nb_iter=iter) # 0.3, 40
        elif str_at == "CnW":
            return attacks.CarliniWagnerL2Attack(self.net, self.n_way, binary_search_steps=9, max_iterations=iter*10) # 9, 10000
        elif str_at == "EAD":
            return attacks.ElasticNetL1Attack(self.net, self.n_way, binary_search_steps=9, max_iterations=iter*10) # 9, 10000
        elif str_at == "DDN":
            return attacks.DDNL2Attack(self.net, nb_iter=iter*3) # 100
        elif str_at == "Single_pixel":
            return attacks.SinglePixelAttack(self.net, max_pixels=iter*3) # 100
        elif str_at == "DeepFool" or str_at == "Deepfool":
            return attacks.DeepfoolLinfAttack(self.net, self.n_way, nb_iter=iter*2, eps=e) # 50, 0.1
        elif str_at == "aRUB":
            return aRUB(self.net, rho=self.rho, q=1, n_way=self.n_way, k_qry=self.k_qry, imgc=self.imgc, imgsz=self.imgsz, device=self.device)
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


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.size(0)
        querysz = x_qry.size(1)

        need_adv = True
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        losses_q_adv = [0 for _ in range(self.update_step + 1)]
        corrects_adv = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct
                

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
                

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] = losses_q[k+1] + loss_q
                
                # adversarial / approximation attack
                if need_adv and k == self.update_step - 1: # for meta-update
                    optimizer.zero_grad()
                    
                    logits_q_adv = self.at.perturb(fast_weights, x_qry[i], y_qry[i])
                    
                    loss_q_adv = F.cross_entropy(logits_q_adv, y_qry[i])
                    losses_q_adv[k + 1] = losses_q_adv[k+1] + loss_q_adv
                

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
                    
                    #PGD AT
                    if need_adv and k == self.update_step - 1:
                        pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                        correct_adv = torch.eq(pred_q_adv, y_qry[i]).sum().item()
                        corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num #마지막 원소 사용 -> K step update한 후!
        
        loss_q_adv = losses_q_adv[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        
        if need_adv:
            self.meta_optim_adv.zero_grad()
            loss_q_adv.backward()
            self.meta_optim_adv.step()
            accs_adv = np.array(corrects_adv) / (querysz * task_num)
        else:
            accs_adv = [0]
        
        return accs, accs_adv, loss_q, loss_q_adv


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        
        need_adv = True
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        corrects_adv = [0 for _ in range(self.update_step_test + 1)]
        corrects_adv_prior = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        
        '''
        # Adversaruak Attack
        if need_adv:
            data = x_qry
            label = y_qry
            optimizer.zero_grad()
            adv_inp_adv = self.test_at.perturb(fast_weights, data, label)
        '''

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            #find the correct index
            corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
            
            
        # Adversarial Attack
        if need_adv:
            data = x_qry
            label = y_qry
            optimizer.zero_grad()
            logits_q_adv = self.test_at.perturb(net.parameters(), data, label)
            with torch.no_grad():
                #logits_q_adv = net(adv_inp, net.parameters(), bn_training=True) # perturb 결과 자체가 logit으로 변경
                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, label).sum().item()
                correct_adv_prior = torch.eq(pred_q_adv[corr_ind], label[corr_ind]).sum().item()
                corrects_adv[0] = corrects_adv[0] + correct_adv
                if len(corr_ind)!=0:
                    corrects_adv_prior[0] = corrects_adv_prior[0] + correct_adv_prior/len(corr_ind)

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            #find the correct index
            corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
            
            
            #PGD AT
            if need_adv:
                #logits_q_adv = net(adv_inp_adv, fast_weights, bn_training=True)
                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, label).sum().item()
                correct_adv_prior = torch.eq(pred_q_adv[corr_ind], label[corr_ind]).sum().item()
                corrects_adv[1] = corrects_adv[1] + correct_adv
                corrects_adv_prior[1] = corrects_adv_prior[1] + correct_adv_prior/len(corr_ind)
            

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            
            # Adversarial Attack
            if need_adv:
                data = x_qry
                label = y_qry
                optimizer.zero_grad()
                logits_q_adv = self.test_at.perturb(fast_weights, data, label)

                #logits_q_adv = net(adv_inp_adv, fast_weights, bn_training=True)
                loss_q_adv = F.cross_entropy(logits_q_adv, label)
        
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                #find the correct index
                corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
                
                # Adversarial Attack
                if need_adv:
                    pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                    correct_adv = torch.eq(pred_q_adv, label).sum().item()
                    correct_adv_prior = torch.eq(pred_q_adv[corr_ind], label[corr_ind]).sum().item()
                    corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv
                    corrects_adv_prior[k + 1] = corrects_adv_prior[k + 1] + correct_adv_prior/len(corr_ind)


        del net

        accs = np.array(corrects) / querysz
        
        accs_adv = np.array(corrects_adv) / querysz
        
        accs_adv_prior = np.array(corrects_adv_prior)

        return accs, accs_adv, accs_adv_prior, loss_q, loss_q_adv
    
    def get_model(self):
        return self.net.state_dict() # save only parameters
        #return self.net # save all model
    
    def set_model(self, model):
        self.net.load_state_dict(model) # if model is only parameters
        #self.net = model # if model is all model

    def set_test_attack(self, attack, eps=2, iter=10):
        self.test_at = self.setAttack(attack, eps/255, iter=iter)

def main():
    pass


if __name__ == '__main__':
    main()

