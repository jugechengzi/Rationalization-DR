import pandas
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

a=[3.01,1.63,3.35,2.03,4.89,2.92]
b=[0.37,0.21,0.42,0.24,0.61,0.37]
c=[0.10,0.05,0.13,0.06,0.18,0.10]

idx=int(len(a)/2)
print(idx)

sr_a=np.array([a[int(2*i)] for i in range(idx)])
print(sr_a)
rnp_a=np.array([a[int(2*i+1)] for i in range(idx)])
print(rnp_a)

sr_b=np.array([b[int(2*i)] for i in range(idx)])
rnp_b=np.array([b[int(2*i+1)] for i in range(idx)])

sr_c=np.array([c[int(2*i)] for i in range(idx)])
rnp_c=np.array([c[int(2*i+1)] for i in range(idx)])



sr=[sr_a,sr_b,sr_c]
rnp=[rnp_a,rnp_b,rnp_c]

index=np.arange(idx)
index_rnp=index+0.2
index_sr=index-0.2
for i in range(3):
    plt.bar(index_rnp,rnp[i],width=0.4)
    plt.bar(index_sr,sr[i],width=0.4)
    plt.xticks(index,['Location','Service','Cleanliness'],fontsize=18)
    plt.yticks(fontsize=12)
    plt.ylabel('Distance',fontsize=15)
    plt.legend(['Random rationale','Gold rationale'],fontsize=15)
    plt.show()

