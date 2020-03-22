import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import re
# d = {    
#     'col1':[1,2,3,4,5,6,7,8,9,10,11,12], 
#     # 'col2':['a','b','np.nan','c','d','e','f'],
#     # 'col3':['sd','asdb','asdas','cgh','fgh','fgh','werf'],
#     # 'col4':[0.2,0.10,0.35,0.52,0.6917,30,0.627]
#     }
# d2 = {
#     'col1':[13,14,15], 
#     # 'col2':['a','b','np.nan','c','d','e','f'],
#     # 'col3':['sd','asdb','asdas','cgh','fgh','fgh','werf'],
#     # 'col4':[0.2,0.10,0.35,0.52,0.6917,30,627]
#     }
# data = pd.DataFrame(data= d)
# data2 = pd.DataFrame(data= d2)
# discretizer = KBinsDiscretizer(n_bins=2, encode='onehot-dense',strategy='quantile')
# result = discretizer.fit_transform(data)
# result2 = discretizer.transform(data2)
# print(result)
# print(result2)
# ages = np.array( [10,20,40,60,120])
# print("age_"+str(ages[np.argmax(ages > 9)]))

my_pattern = r".*\(.*\).*"
my_string = "aaa)aaaa)"
result = re.match(my_pattern,my_string)
print(result)


