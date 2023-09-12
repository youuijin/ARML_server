import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

import math

class loss_function():
    def __init__(self, alpha=0.05, beta=1.0, zeta=10):
        #print("initialize lambda")
        self.alpha = alpha # for AT
        self.beta = beta # for trades
        self.zeta = zeta # for WAR
        # self.lamb = 0
        self.loss = self.standard_loss
        
    def set_loss(self, is_qry, loss_str):
        if not is_qry:
            return self.standard_loss
        if(loss_str=="no"):
            return self.standard_loss
        elif(loss_str=="AT"):
            return self.AT_loss
        elif(loss_str=="trades"):
            return self.trades_loss
        elif(loss_str=="AT-WAR"):
            return self.AT_WAR_loss
        elif(loss_str=="trades-WAR"):
            return self.trades_WAR_loss
        else:
            print("Wrong Type Loss")
            exit()

    # return : loss(include grad), loss, loss_clean, loss_adv    
    def standard_loss(self, model, fast_weights, x, y, at, _):
        logits = model(x, fast_weights, bn_training=True)
        loss = F.cross_entropy(logits, y)
        return loss, loss.item(), loss.item(), 0, 0

    def AT_loss(self, model, fast_weights, x, y, at, _):
        logits = model(x, fast_weights, bn_training=True)
        loss_clean = F.cross_entropy(logits, y)
        
        logits_adv = at.perturb(fast_weights, x, y)
        loss_adv = F.cross_entropy(logits_adv, y)

        loss = loss_clean + self.alpha* loss_adv

        return loss, loss.item(), loss_clean.item(), loss_adv.item(), 0

    def trades_loss(self, model, fast_weights, x, y, at, _):
        criterion_kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        epsilon = 1e-8

        logits = model(x, fast_weights, bn_training=True) # TODO : 다른 모델도 bn_training이라는 인자를 받을 수 있어야 함
        logits_adv = at.perturb(fast_weights, x, y)
        loss_clean = F.cross_entropy(logits, y)
        loss_adv = (1.0/len(x))*criterion_kl(torch.log(F.softmax(logits_adv, dim=1) + epsilon), torch.log(F.softmax(logits, dim=1) + epsilon))
        
        loss = loss_clean + self.beta * loss_adv

        return loss, loss.item(), loss_clean.item(), loss_adv.item(), 0
    

    def AT_WAR_loss(self, model, fast_weights, x, y, at, lamb):
        alpha = 0.1
        epsilon = 1e-8
        logits = model(x, fast_weights, bn_training=True)
        logits_adv = at.perturb(fast_weights, x, y)

        loss_clean = F.cross_entropy(logits, y)
        loss_adv = F.cross_entropy(logits_adv, y)
        # loss_adv_item = loss_adv.item()
        if(loss_adv>=loss_clean):
            loss_robust = loss_adv - F.cross_entropy(logits, y)
        else:
            loss_robust = torch.tensor(0).to(logits.device)
        
        #loss_adv = max(F.cross_entropy(logits_adv, y) - F.cross_entropy(logits, y),0)

        lamb = lamb + alpha*(self.zeta-(loss_clean.item()/(loss_robust.item()+epsilon)))
        lamb = max(lamb, 0)
        #print(self.lamb)
        if(math.isnan(lamb)):
            exit()

        loss = loss_clean + lamb * loss_robust

        return loss, loss.item(), loss_clean.item(), loss_adv.item(), lamb

    def trades_WAR_loss(self, model, fast_weights, x, y, at, lamb):
        alpha = 0.1
        epsilon = 1e-8
        criterion_kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

        logits = model(x, fast_weights, bn_training=True)
        logits_adv = at.perturb(fast_weights, x, y)

        loss_clean = F.cross_entropy(logits, y)
        loss_robust = (1.0/len(x))*criterion_kl(torch.log(F.softmax(logits_adv, dim=1) + epsilon), torch.log(F.softmax(logits, dim=1) + epsilon))

        # if(loss_adv>=loss_clean):
        #     loss_robust = loss_adv - F.cross_entropy(logits, y)
        # else:
        #     loss_robust = torch.tensor(0).to(logits.device)

        lamb = lamb + alpha*(self.zeta-(loss_clean.item()/(loss_robust.item()+epsilon)))
        lamb = max(lamb, 0)
        #self.lamb = np.clip(lamb, 0, 1)
        #print(self.lamb)
        if(math.isnan(lamb)):
            exit()

        loss = loss_clean + lamb * loss_robust

        return loss, loss.item(), loss_clean.item(), loss_robust.item(), lamb