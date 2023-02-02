import pandas
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

a=[16.42,33.92,	24.43,	42.6,	26,	55.4,	22.05,	47.48,	26.92,	45.15,	26.66,	197.2]
b=[1.88,	4.44,	2.67,	5.05,	2.72,	6.5	,3.49,	7.1,3.07,	7.19,	4.58,	18.06]
c=[1.04,	2.68,	1.13,	2.76,	2.09,	3.48,	1.72,	3.33,	1.45,	4.13,	2.05,	7.98]
d=[0.7,	1.75,	0.82,	1.78,	1.37,	2.87,	1.18,	2.67,	1.02,	2.5,	1.34,	5.9]
e=[0.51,	1.6,	0.68,	1.28,	0.9	,1.7	,1.02,	1.51,	0.81,	1.4,	1.02,	3.91]
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

sr_d=np.array([d[int(2*i)] for i in range(idx)])
rnp_d=np.array([d[int(2*i+1)] for i in range(idx)])

sr_e=np.array([e[2*i] for i in range(idx)])
rnp_e=np.array([e[2*i+1] for i in range(idx)])

sr=[sr_a,sr_b,sr_c,sr_d,sr_e]
rnp=[rnp_a,rnp_b,rnp_c,rnp_d,rnp_e]

index=np.arange(idx)
index_rnp=index+0.2
index_sr=index-0.2
for i in range(5):
    plt.bar(index_rnp,rnp[i],width=0.4)
    plt.bar(index_sr,sr[i],width=0.4)
    plt.xticks(index,['Appearance','Aroma','Palate','Location','service','Cleanliness'],fontsize=11)
    plt.yticks(fontsize=12)
    plt.ylabel('Lc',fontsize=15)
    plt.legend(['RNP','SR'],fontsize=15)
    plt.show()

