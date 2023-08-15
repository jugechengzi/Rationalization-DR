# DR: Decoupled Rationalization
This repo contains Pytorch implementation of [Decoupled Rationalization with Asymmetric Learning Rates: A Flexible Lipschitz Restraint (KDD 2023)](https://github.com/jugechengzi/Rationalization-DR/blob/main/paper.pdf).

**The bibtex in Google Scholar is incorrect, please use the citation we provide below.**

**We have provided some tips for you to better understand our code and build your method on top of it: [tips](https://github.com/jugechengzi/Rationalization-DR/edit/main/tips.pdf)** If you're still having trouble reproducing your code after reading these tips, I'd be happy to set up a video meeting to help you.

This paper is greatly appreciated by some of the reviewers from ACL ARR October and KDD. See [reviews](https://github.com/jugechengzi/Rationalization-DR/blob/main/REVIEW.pdf).

We would be grateful if you would star this repo before cloning it.

If you have any questions, please feel free to open an issue or just send us an email.

## Abstract
In this paper, we theoretically bridge the degeneration problem with the predictor’s Lipschitz continuity. Then, we emperically propose a simple but effective method named DR, which can naturally
and flexibly constrain the Lipschitz constant of the predictor, to address the problem of degeneration. The main idea of DR is to decouple the generator and predictor to allocate them with asymmetric learning rates.

## Environments
torch 1.12.0+cu113. 

python 3.7.13. 

RTX3090

We suggest you to create a new environment with: conda create -n DR python=3.7.13

Then activate the environment with: conda activate DR

Install pytorch: conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

Install other packages: pip install -r requirements.txt






## Datasets
Beer Reviews: you can get it [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel Reviews: you can get it [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.

## Running example
### Beer
Aroma: python -u norm_beer.py --lr 0.0002 --batch_size 128 --gpu 0 --sparsity_percentage 0.138 --sparsity_lambda 11 --continuity_lambda 12 --epochs 200 --aspect 1


### Hotel  
Service: python -u norm_beer.py --data_type hotel --lr 0.0001 --batch_size 1024 --gpu 0 --sparsity_percentage 0.115 --sparsity_lambda 10 --continuity_lambda 10 --epochs 400 --aspect 1

**_Notes_**: "--sparsity_percentage 0.138" means "$s=0.138$" in Eq.3 (But the actual sparsity is different from $s$). "--sparsity_lambda 11 --continuity_lambda 12 " means $\lambda_1=11, \lambda_2=12$. "--epochs 200" means we run 200 epochs and take the results when the "dev_acc" is best.

## Result  
You will get a result like "best_dev_epoch=184" at last. Then you need to find the result corresponding to the epoch with number "184".  
For Beer-Aroma, you may get a result like: 

Train time for epoch #184 : 7.447731 second  
gen_lr=0.0002, pred_lr=3.7614679336547854e-05  
traning epoch:184 recall:0.8173 precision:0.8665 f1-score:0.8412 accuracy:0.8457  
Validate  
dev epoch:184 recall:0.8194 precision:0.9261 f1-score:0.8695 accuracy:0.8161  
Validate Sentence  
dev dataset : recall:0.9792 precision:0.8017 f1-score:0.8816 accuracy:0.8033  
Annotation  
annotation dataset : recall:0.8703 precision:0.9973 f1-score:0.9295 accuracy:0.8723  
The annotation performance: sparsity: 15.6389, precision: 77.1722, recall: 77.4658, f1: 77.3188  
Annotation Sentence  
annotation dataset : recall:0.9835 precision:0.9858 f1-score:0.9847 accuracy:0.9704  
Rationale  
rationale dataset : recall:0.8738 precision:0.9920 f1-score:0.9292 accuracy:0.8712  

The line "annotation dataset : recall:0.8703 precision:0.9973 f1-score:0.9295 accuracy:0.8723" indicates that the prediction accuracy on the test set is 87.23. And the line 
"The annotation performance: sparsity: 15.6389, precision: 77.1722, recall: 77.4658, f1: 77.3188" indicates that the rationale F1 score is 77.3188.

## Citation


@inproceedings{10.1145/3580305.3599299,  
author = {Liu, Wei and Wang, Jun and Wang, Haozhao and Li, Ruixuan and Qiu, Yang and Zhang, YuanKai and Han, Jie and Zou, Yixiong},  
title = {Decoupled Rationalization with Asymmetric Learning Rates: A Flexible Lipschitz Restraint},  
year = {2023},  
isbn = {9798400701030},  
publisher = {Association for Computing Machinery},  
address = {New York, NY, USA},  
url = {https://doi.org/10.1145/3580305.3599299},  
doi = {10.1145/3580305.3599299},  
booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},  
pages = {1535–1547},  
numpages = {13},  
keywords = {cooperative game, lipschitz continuity, interpretability},  
location = {Long Beach, CA, USA},  
series = {KDD '23}  
}












