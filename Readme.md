# FR: Folded Rationalization with a Unified Encoder
This repo contains Pytorch implementation of Decoupled Rationalization (DR).  
## Environments
torch 1.10.2+cu113. python 3.7.9. tensorboardx 2.4. tensorboard 2.6.0
## Datasets
Beer Reviews: you can get it [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel Reviews: you can get it [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.

## Running example
### Beer
Appearance: python -u norm_beer.py --dis_lr 1 --hidden_dim 200 --data_type beer --save 0 --dropout 0.2 --lr 0.0001 --batch_size 128 --gpu 1 --sparsity_percentage 0.163 --sparsity_lambda 11 --continuity_lambda 12 --epochs 200 --aspect 0 








