import argparse
import os
import time

import torch
import math
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from beer import BeerData, BeerAnnotation
from hotel import HotelData,HotelAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import Sp_norm_model

from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter


def parse():
    #默认： nonorm, dis_lr=1, data=beer, save=0
    parser = argparse.ArgumentParser(
        description="Distribution Matching Rationalization")

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--sp_norm',
                        type=int,
                        default=0,
                        help='0:rnp,1:norm')
    parser.add_argument('--dis_lr',
                        type=int,
                        default=1,
                        help='0:rnp,1:dis')
    parser.add_argument('--aspect',
                        type=int,
                        default=2,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')

    # model parameters
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.1,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args(args=[])
    return args


#####################
# set random seed
#####################
# torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)

######################
# load embedding
######################
pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
if args.data_type=='beer':       #beer
    train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

    dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)

    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
elif args.data_type == 'hotel':       #hotel
    args.data_dir='./data/hotel'
    args.annotation_path='./data/hotel/annotations'
    train_data = HotelData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

    dev_data = HotelData(args.data_dir, args.aspect, 'dev', word2idx)

    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
writer=SummaryWriter(args.writer)
model=Sp_norm_model(args)
model.to(device)

######################
# Training
######################
g_para=list(map(id, model.generator.parameters()))
p_para=filter(lambda p: id(p) not in g_para, model.parameters())
lr2=args.lr
lr1=args.lr
para=[
    {'params': model.generator.parameters(), 'lr':lr1},
    {'params':p_para,'lr':lr2}
]
optimizer = torch.optim.Adam(para)
print('lr1={},lr2={}'.format(lr1,lr2))
# optimizer = torch.optim.Adam(model.parameters())

######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]


def get_grad(model,dataloader,p,use_rat=1):
    data=0
    model.to(device)
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



# rat_grad_2=get_grad(model,train_loader,2,1)
# rat_rnp_2=get_grad(rnp,train_loader,2,1)
# rat_rnp_inf=get_grad(rnp,train_loader,float('inf'),1)
# rat_grad_inf=get_grad(model,train_loader,float('inf'),1)
# rat_grad_1=get_grad(model,train_loader,1,1)
# rat_rnp_1=get_grad(rnp,train_loader,1,1)
# rat_rnp_3=get_grad(rnp,train_loader,3,1)
# rat_grad_3=get_grad(model,train_loader,3,1)
# rat_grad_4=get_grad(model,train_loader,4,1)
# rat_rnp_4=get_grad(rnp,train_loader,4,1)
