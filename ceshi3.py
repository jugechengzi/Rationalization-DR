import argparse
import os
import time

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from beer import BeerData, BeerAnnotation
from hotel import HotelData,HotelAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import Sp_norm_model,GenEncNoShareModel

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
                        default='hotel',
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

model=GenEncNoShareModel(args)




x_text=annotation_data.input_ids
mask=annotation_data.masks
pos_text=[]
neg_text=[]
pos_rat=[]
neg_rat=[]
pos_mask=[]
neg_mask=[]
rat=annotation_data.rationales
rat_len=torch.sum(rat,dim=1)
sent_len=torch.sum(mask,dim=1)
sent_norat=[]
for idx,i in enumerate(rat_len):
    if i<10:
        sent_norat.append(idx)
    elif i/sent_len[idx]>0.5:
        sent_norat.append(idx)
for idx,i in enumerate(annotation_data.labels):
    if idx in sent_norat:
        continue
    if i==0:
        neg_text.append(x_text[idx].unsqueeze(0))
        neg_rat.append(rat[idx].unsqueeze(0))
        neg_mask.append(annotation_data.masks[idx].unsqueeze(0))
    elif i==1:
        pos_text.append(x_text[idx].unsqueeze(0))
        pos_rat.append(rat[idx].unsqueeze(0))
        pos_mask.append(annotation_data.masks[idx].unsqueeze(0))

neg_rat=torch.cat(neg_rat,dim=0)
neg_text=torch.cat(neg_text,dim=0)
neg_mask=torch.cat(neg_mask,dim=0)
pos_rat=torch.cat(pos_rat,dim=0)
pos_text=torch.cat(pos_text,dim=0)
pos_mask=torch.cat(pos_mask,dim=0)


def get_rand_rat(text,mask,rat):
    rat_len=torch.sum(rat,dim=1)
    no_rat=1-rat
    rand_mat=torch.rand_like(text.float())*mask*no_rat
    sample_rat=[]
    uninformative_rat=torch.zeros_like(rat)
    for idx,text_i in enumerate(rand_mat):
        temp_r=torch.multinomial(text_i,rat_len[idx].item(),replacement=False)
        for j in temp_r:
            uninformative_rat[idx,j]=1
    return uninformative_rat

def get_avg_emb(text,rat):
   text_emb=model.embedding_layer(text)
   rat_emb=text_emb*rat.unsqueeze(-1)
   sum_emb=torch.sum(rat_emb,dim=1)
   avg_emb=sum_emb/torch.sum(rat,dim=1).unsqueeze(-1)       #batch*100
   all_sent=torch.mean(avg_emb,dim=0)                       #100
   return avg_emb,all_sent


def get_trained_emb(text,rat):
    text_emb=model.embedding_layer(text)
    rat_emb = text_emb * rat.unsqueeze(-1)
    trained_emb,_=model.cls(rat_emb)* rat.unsqueeze(-1)
    sum_emb = torch.sum(trained_emb, dim=1)
    avg_emb = sum_emb / torch.sum(rat, dim=1).unsqueeze(-1)  # batch*100
    all_sent = torch.mean(avg_emb, dim=0)  # 100
    return avg_emb, all_sent

def get_max_emb(text,rat):
    text_emb = model.embedding_layer(text)
    rat_emb = text_emb * rat.unsqueeze(-1)+ (1.-rat.unsqueeze(-1)) * (-1e6)
    rat_emb = torch.transpose(rat_emb, 1, 2)
    max_emb, _ = torch.max(rat_emb, axis=2)     #batch*100
    return max_emb


def get_norm(emb1,emb2,p):
    max_len=min(len(emb1),len(emb2))
    e=abs(emb1-emb2)
    norm=torch.linalg.norm(e,ord=p,dim=0)
    return norm.item()


pos_uninfor_rat=get_rand_rat(pos_text,pos_mask,pos_rat)
neg_uninfor_rat=get_rand_rat(neg_text,neg_mask,neg_rat)


#full text
fupos_emb,fupos_batch=get_avg_emb(pos_text,pos_mask)
funeg_emb,funeg_batch=get_avg_emb(neg_text,neg_mask)
fuinformative_norm_1=get_norm(fupos_batch,funeg_batch,1)
fuinformative_norm_2=get_norm(fupos_batch,funeg_batch,2)
fuinformative_norm_3=get_norm(fupos_batch,funeg_batch,3)
fuinformative_norm_4=get_norm(fupos_batch,funeg_batch,4)
fuinformative_norm_inf=get_norm(fupos_batch,funeg_batch,float('inf'))


#informative:
pos_emb,pos_batch=get_avg_emb(pos_text,pos_rat)
neg_emb,neg_batch=get_avg_emb(neg_text,neg_rat)
informative_norm_1=get_norm(pos_batch,neg_batch,1)
informative_norm_2=get_norm(pos_batch,neg_batch,2)
informative_norm_3=get_norm(pos_batch,neg_batch,3)
informative_norm_4=get_norm(pos_batch,neg_batch,4)
informative_norm_inf=get_norm(pos_batch,neg_batch,float('inf'))

#uninformative
unpos_emb,unpos_batch=get_avg_emb(pos_text,pos_uninfor_rat)
unneg_emb,unneg_batch=get_avg_emb(neg_text,neg_uninfor_rat)
uninformative_norm_1=get_norm(unpos_batch,unneg_batch,1)
uninformative_norm_2=get_norm(unpos_batch,unneg_batch,2)
uninformative_norm_3=get_norm(unpos_batch,unneg_batch,3)
uninformative_norm_4=get_norm(unpos_batch,unneg_batch,4)
uninformative_norm_inf=get_norm(unpos_batch,unneg_batch,float('inf'))
print([(fuinformative_norm_1,informative_norm_1,uninformative_norm_1),(fuinformative_norm_2,informative_norm_2,uninformative_norm_2),(fuinformative_norm_3,informative_norm_3,uninformative_norm_3),(fuinformative_norm_4,informative_norm_4,uninformative_norm_4),(fuinformative_norm_inf,informative_norm_inf,uninformative_norm_inf)])



# #plot
# fig, ax = plt.subplots()
# x=np.linspace(1,41,40)
# ax.plot(x,uninformative_norm_2[:40],label='uninf')
# ax.plot(x,informative_norm_2[:40],label='inf')
# ax.legend(fontsize=14,loc='lower right')
# plt.show()
#
# fig, ax = plt.subplots()
# x=np.linspace(1,41,40)
# ax.plot(x,uninformative_norm_1[:40],label='uninf_norm1')
# ax.plot(x,informative_norm_1[:40],label='inf_norm1')
# ax.legend(fontsize=14,loc='lower right')
# plt.show()
#
#
#
# #informative:
# pos_emb=get_max_emb(pos_text,pos_rat)
# neg_emb=get_max_emb(neg_text,neg_rat)
# informative_norm_1=get_norm(pos_emb,neg_emb,1)
# informative_norm_2=get_norm(pos_emb,neg_emb,2)
#
# #uninformative
# unpos_emb=get_max_emb(pos_text,pos_uninfor_rat)
# unneg_emb=get_max_emb(neg_text,neg_uninfor_rat)
# uninformative_norm_1=get_norm(unpos_emb,unneg_emb,1)
# uninformative_norm_2=get_norm(unpos_emb,unneg_emb,2)
#
#
#
# #plot
# fig, ax = plt.subplots()
# x=np.linspace(1,41,40)
# ax.plot(x,uninformative_norm_2[:40],label='uninf')
# ax.plot(x,informative_norm_2[:40],label='inf')
# ax.legend(fontsize=14,loc='lower right')
# plt.show()
#
# fig, ax = plt.subplots()
# x=np.linspace(1,41,40)
# ax.plot(x,uninformative_norm_1[:40],label='uninf_norm1')
# ax.plot(x,informative_norm_1[:40],label='inf_norm1')
# ax.legend(fontsize=14,loc='lower right')
# plt.show()