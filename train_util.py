import torch
import torch.nn.functional as F

from metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np
import math



def train_noshare(model, optimizer, dataset, device, args,writer_epoch,grad):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss


        #see grad
        l_logits=torch.mean(logits)
        l_logits.backward(retain_graph=True)
        for k,v in model.gen.named_parameters():
            if k == "weight_ih_l0":
                g=abs(v.grad.clone().detach())
                grad.append(g)
        optimizer.zero_grad()
        improve=torch.mean((grad[-1]-grad[0])/grad[0])
        writer_epoch[0].add_scalar('grad', improve, writer_epoch[1]*len(dataset)+batch)



        # update gradient
        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def train_sp_norm(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient
        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr

        # l_logits = torch.mean(logits)
        # l_logits.backward(retain_graph=True)
        # for k, v in model.gen.named_parameters():
        #     if k == "weight_ih_l0":
        #         g = abs(v.grad.clone().detach().cpu())
        #         grad.append(g)
        # optimizer.zero_grad()
        # improve = torch.mean((grad[-1] - grad[0]) / grad[0])
        # writer_epoch[0].add_scalar('grad', improve, writer_epoch[1] * len(dataset) + batch)
        #
        # cls_loss.backward(retain_graph=True)
        # for k, v in model.gen.named_parameters():
        #     if k == "weight_ih_l0":
        #         g = abs(v.grad.clone().detach().cpu())
        #         grad_loss.append(g)
        # optimizer.zero_grad()
        # improve2 = torch.mean((grad_loss[-1] - grad_loss[0]) / grad_loss[0])
        # writer_epoch[0].add_scalar('grad_loss', improve2, writer_epoch[1] * len(dataset) + batch)

        loss.backward()
        # a1=model.gen.weight_hh_l0.grad[:100,:].view(-1)
        # a2 = model.gen.weight_hh_l0.grad[100:200, :].view(-1)
        # a3 = model.gen.weight_hh_l0.grad[200:300, :].view(-1)
        # a4=model.gen_fc.weight.grad[1,:].view(-1)
        # b1 = model.cls.weight_hh_l0.grad[:100, :].view(-1)
        # b2 = model.cls.weight_hh_l0.grad[100:200, :].view(-1)
        # b3 = model.cls.weight_hh_l0.grad[200:300, :].view(-1)
        # b4 = model.cls_fc.weight.grad[1, :].view(-1)

        a1 = model.gen.weight_ih_l0.grad[:100, :].view(-1)
        a2 = model.gen.weight_ih_l0.grad[100:200, :].view(-1)
        a3 = model.gen.weight_ih_l0.grad[200:300, :].view(-1)
        a4 = model.gen_fc.weight.grad[1, :].view(-1)
        b1 = model.cls.weight_ih_l0.grad[:100, :].view(-1)
        b2 = model.cls.weight_ih_l0.grad[100:200, :].view(-1)
        b3 = model.cls.weight_ih_l0.grad[200:300, :].view(-1)
        # b4 = model.cls_fc.weight.grad[1, :].view(-1)

        writer_epoch[0].add_scalars('a1_p=2', {'gen': torch.linalg.norm(a1, ord=2, dim=0)}, (writer_epoch[1])*batch_len+batch)
        writer_epoch[0].add_scalars('a1_p=2', {'pred': torch.linalg.norm(b1, ord=2, dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a2_p=2', {'gen': torch.linalg.norm(a2, ord=2, dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a2_p=2', {'pred': torch.linalg.norm(b2, ord=2, dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a3_p=2', {'gen': torch.linalg.norm(a3, ord=2, dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a3_p=2', {'pred': torch.linalg.norm(b3, ord=2, dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a4_p=2', {'gen': torch.linalg.norm(a4, ord=2, dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        # writer_epoch[0].add_scalars('a4_p=2', {'pred': torch.linalg.norm(b4, ord=2, dim=0)},
        #                             (writer_epoch[1]) * batch_len + batch)

        writer_epoch[0].add_scalars('a1_p=inf', {'gen': torch.linalg.norm(a1, ord=float('inf'), dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a1_p=inf', {'pred': torch.linalg.norm(b1, ord=float('inf'), dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a2_p=inf', {'gen': torch.linalg.norm(a2, ord=float('inf'), dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a2_p=inf', {'pred': torch.linalg.norm(b2, ord=float('inf'), dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a3_p=inf', {'gen': torch.linalg.norm(a3, ord=float('inf'), dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a3_p=inf', {'pred': torch.linalg.norm(b3, ord=float('inf'), dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        writer_epoch[0].add_scalars('a4_p=inf', {'gen': torch.linalg.norm(a4, ord=float('inf'), dim=0)},
                                    (writer_epoch[1]) * batch_len + batch)
        # writer_epoch[0].add_scalars('a4_p=inf', {'pred': torch.linalg.norm(b4, ord=float('inf'), dim=0)},
        #                             (writer_epoch[1]) * batch_len + batch)


        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)
    writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    return precision, recall, f1_score, accuracy

def classfy(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        logits = model(inputs, masks)

        # computer loss
        cls_loss =F.cross_entropy(logits, labels)


        loss = cls_loss

        # update gradient
        loss.backward()
        print('yes')
        optimizer.step()
        print('yes2')

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

# def get_grad(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
#     TP = 0
#     TN = 0
#     FN = 0
#     FP = 0
#     cls_l = 0
#     spar_l = 0
#     cont_l = 0
#     train_sp = []
#     for (batch, (inputs, masks, labels)) in enumerate(dataset):
#         optimizer.zero_grad()
#
#         inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
#
#         # rationales, cls_logits
#         rationales, logits = model(inputs, masks)
#
#         # computer loss
#         cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
#
#         sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
#             rationales[:, :, 1], masks, args.sparsity_percentage)
#
#         sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
#         train_sp.append(
#             (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())
#
#         continuity_loss = args.continuity_lambda * get_continuity_loss(
#             rationales[:, :, 1])
#
#         loss = cls_loss + sparsity_loss + continuity_loss
#         # update gradient
#         if args.dis_lr == 1:
#             if sparsity == 0:
#                 lr_lambda = 1
#             else:
#                 lr_lambda = sparsity
#             if lr_lambda < 0.05:
#                 lr_lambda = 0.05
#             optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
#         elif args.dis_lr == 0:
#             pass
#         else:
#             optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr
#
#         l_logits = torch.mean(logits)
#         l_logits.backward(retain_graph=True)
#         for k, v in model.gen.named_parameters():
#             if k == "weight_ih_l0":
#                 g = abs(v.grad.clone().detach().cpu())
#                 grad.append(g)
#         optimizer.zero_grad()
#         improve = torch.mean((grad[-1] - grad[0]) / grad[0])
#         writer_epoch[0].add_scalar('grad', improve, writer_epoch[1] * len(dataset) + batch)
#
#         cls_loss.backward(retain_graph=True)
#         for k, v in model.gen.named_parameters():
#             if k == "weight_ih_l0":
#                 g = abs(v.grad.clone().detach().cpu())
#                 grad_loss.append(g)
#         optimizer.zero_grad()
#         improve2 = torch.mean((grad_loss[-1] - grad_loss[0]) / grad_loss[0])
#         writer_epoch[0].add_scalar('grad_loss', improve2, writer_epoch[1] * len(dataset) + batch)
#
#         loss.backward()
#         optimizer.step()
#
#         cls_soft_logits = torch.softmax(logits, dim=-1)
#         _, pred = torch.max(cls_soft_logits, dim=-1)
#
#         # TP predict 和 label 同时为1
#         TP += ((pred == 1) & (labels == 1)).cpu().sum()
#         # TN predict 和 label 同时为0
#         TN += ((pred == 0) & (labels == 0)).cpu().sum()
#         # FN predict 0 label 1
#         FN += ((pred == 0) & (labels == 1)).cpu().sum()
#         # FP predict 1 label 0
#         FP += ((pred == 1) & (labels == 0)).cpu().sum()
#         cls_l += cls_loss.cpu().item()
#         spar_l += sparsity_loss.cpu().item()
#         cont_l += continuity_loss.cpu().item()
#
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1_score = 2 * recall * precision / (recall + precision)
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
#     writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
#     writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
#     writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
#     return precision, recall, f1_score, accuracy

def train_g_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.g_skew(inputs,masks)[:,0,:]
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def get_grad(model,dataloader,p,use_rat,device):
    data=0
    # device=model.device()
    model.train()
    grad=[]
    for batch,d in enumerate(dataloader):
        data=d
        inputs, masks, labels = data
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        rationale,logit,embedding2,cls_embed=model.grad(inputs, masks)
        loss=torch.mean(torch.softmax(logit,dim=-1)[:,1])
        cls_embed.retain_grad()
        loss.backward()
        if use_rat==0:
            k_mask=masks
        elif use_rat==1:
            k_mask=rationale[:,:,1]
        masked_grad=cls_embed.grad*k_mask.unsqueeze(-1)
        gradtemp=torch.sum(abs(masked_grad),dim=1)       #bat*256*100→bat*100,在句子长度方向相加
        gradtemp=gradtemp/torch.sum(k_mask,dim=-1).unsqueeze(-1)      #在句子长度方向取平均
        # gradtempmask=gradtemp*rationale[:,:,1]
        # gradtempmaskmean =torch.sum(gradtempmask,dim=-1)/torch.sum(rationale[:,:,1],dim=-1)    #在句子长度方向取平均
        gradtempmask = gradtemp
        norm_grad=torch.linalg.norm(gradtempmask, ord=p, dim=1)           #在维度上取norm
        # gradtempmaskmean = torch.sum(gradtempmask, dim=-1) / torch.sum(masks, dim=-1)  # 在句子长度方向取平均
        grad.append(norm_grad.clone().detach().cpu())
    grad=torch.cat(grad,dim=0)
    tem=[]
    for g in grad:
        if math.isnan(g.item()):
            continue
        else:
            tem.append(g)

    tem=torch.tensor(tem)
    maxg=torch.max(tem)*1000
    meang=torch.mean(tem)*1000
    return maxg,meang


