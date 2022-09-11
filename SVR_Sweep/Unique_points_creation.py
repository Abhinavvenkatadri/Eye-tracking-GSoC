import numpy as np
import pandas as pd
import random

Main_Collection = pd.read_csv('/home/abhinav1/Users_MIT/CSVs_MIT_Dinesh/03183.csv')

IDs = ['3183']

data = []

random.seed(22)

for x in IDs:
    part_ind = Main_Collection[Main_Collection['Image_ID'].str.startswith(x)]
    
    main_dict = {}
    
    for ind in part_ind.index:
        if part_ind['GT_Value'][ind] not in main_dict.keys():
            main_dict[part_ind['GT_Value'][ind]] = [ind]
        else:
            main_dict[part_ind['GT_Value'][ind]].append(ind)
    
    temp_list = []
    
    for i,x in enumerate(main_dict):
        temp_list.append(random.choice(main_dict[x]))
        
    for x in temp_list:
        data.append([part_ind['Image_ID'][x],part_ind['Penultimate_Output'][x],part_ind['GT_Value'][x]])
        
df=pd.DataFrame(data,columns=['Image_ID','Penultimate_Output','GT_Value'])
path_csv = '/home/abhinav1/Users_MIT/CSVs_MIT/unique30_3183.csv'
df.to_csv(path_csv, index = False)