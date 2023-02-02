# FR: Folded Rationalization with a Unified Encoder
This repo contains Pytorch implementation of Folded Rationalization (FR).
##Environments
torch 1.10.2+cu113. python 3.7.9. tensorboardx 2.4. tensorboard 2.6.0
## Datasets
Beer Reviews: you can get it from [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel Reviews: you can get it from [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.

## Running example
### Beer
Appearance: python -u norm_beer.py --dis_lr 1 --hidden_dim 200 --data_type beer --save 0 --dropout 0.2 --lr 0.0001 --batch_size 128 --gpu 1 --sparsity_percentage 0.163 --sparsity_lambda 11 --continuity_lambda 12 --epochs 200 --aspect 0 





## Result
You will get a result like "best_dev_epoch=165" at last. Then you need to find the result corresponding to the epoch with number "165".  
For Beer-Appearance, you may get a result like:  
Train time for epoch #165 : 6.419411 second  
gen_lr=0.0001, pred_lr=1.664898693561554e-05  
traning epoch:165 recall:0.8007 precision:0.8592 f1-score:0.8289 accuracy:0.8347  
Validate  
dev epoch:165 recall:0.8164 precision:0.9365 f1-score:0.8723 accuracy:0.8186  
Annotation  
annotation dataset : recall:0.8516 precision:0.9987 f1-score:0.9193 accuracy:0.8526  
The annotation performance: sparsity: 18.5362, precision: 82.4436, recall: 82.5414, f1: 82.4925  
 
The last line indicates the overlap between the selected tokens and human-annotated  rationales. The penultimate line shows the predictive accuracy on the test set.  



