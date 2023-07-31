import advertorchMeta.attacks as attacks
from aRUBattack import aRUB

# 수정 필요 - class 로 되어 있음

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
        return attacks.CarliniWagnerL2Attack(self.net, self.n_way, binary_search_step=9, max_iterations=10000) # 9, 10000
    elif str_at == "EAD":
        return attacks.ElasticNetL1Attack(self.net, self.n_way, binary_search_steps=9, max_iterations=10000) # 9, 10000
    elif str_at == "DDN":
        return attacks.DDNL2Attack(self.net, nb_iter=iter) # 100
    elif str_at == "Single_pixel":
        return attacks.SinglePixelAttack(self.net, max_pixels=iter) # 100
    elif str_at == "DeepFool" or str_at == "Deepfool":
        return attacks.DeepfoolLinfAttack(self.net, self.n_way, nb_iter=iter, eps=e) # 50, 0.1
    elif str_at == "aRUB":
        return aRUB(self.net, rho=self.rho, q=1, n_way=self.n_way, k_qry=self.k_qry, imgc=self.imgc, imgsz=self.imgsz, device=self.device)
    
    else:
        print("wrong type Attack")
        exit()