import numpy as np
from collections import defaultdict
import pandas as pd
import ipdb

data_path = 'train_test_data.xlsx'
xl_file = pd.ExcelFile(data_path)
df_names= [('bin1_train','bin1_test'),('bin2_train','bin2_test'),('bin3_train','bin3_test')]
dfs = {sheet_name: xl_file.parse(sheet_name) 
	for sheet_name in xl_file.sheet_names}

train_data1 = dfs[df_names[0][0]][['userId','movieId','rating']].to_numpy().astype(int)


print('STARTING')
def make_watch_dict(data):
	watch_dictionary = defaultdict(list)
	for i in range(len(data)):
		user_id = data[i,0]
		movieId = data[i,1]
		rating = data[i,2]
		watch_dictionary[user_id].append((movieId, rating))
	return watch_dictionary

w = make_watch_dict(train_data1)
ipdb.set_trace()


