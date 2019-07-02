import pandas as pd
import pickle
# 数据准备 提速
action_path = "./data/jdata_action.csv"
comment_path = "./data/jdata_comment.csv"
product_path = "./data/jdata_product.csv"
user_path = "./data/jdata_user.csv"
shop_path = "./data/jdata_shop.csv"

user = pd.read_csv(user_path, sep=',')
product = pd.read_csv(product_path, sep=',')
action = pd.read_csv(action_path, sep=',')
comment = pd.read_csv(comment_path, sep=',')
shop = pd.read_csv(shop_path, sep=',')

pickle.dump(user, open('./cache/origin_user.pkl', 'wb'))
pickle.dump(product, open('./cache/origin_product.pkl', 'wb'))
pickle.dump(action, open('./cache/origin_action.pkl', 'wb'))
pickle.dump(comment, open('./cache/origin_comment.pkl', 'wb'))
pickle.dump(shop, open('./cache/origin_shop.pkl', 'wb'))

"""
F12_7
"""
from user_cate_shop import make_train_set_F12_7, make_test_set_F12_7,lgb_train_F12_7

ignore_feat = ['label', 'type', 'user_id', 'cate','shop_id', 'sku_id','action_time', 'dt', 'market_time', 'shop_reg_tm',
               'user_reg_tm', 'vender_id', 'module_id']

label_start_date = '2018-04-09'
label_end_date = '2018-04-16'
train_start_date = '2018-03-08'
train_end_date = '2018-04-09'
test_start_date = '2018-03-15'
test_end_date = '2018-04-16'

training_data = make_train_set_F12_7(train_start_date, train_end_date, label_start_date, label_end_date, start='2018-02-01')
sub_training_data = make_test_set_F12_7(test_start_date, test_end_date, start='2018-02-01')

# train
feats = [f for f in training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
label = training_data['label'].copy()
train = training_data[feats].copy()

# test
feats = [f for f in sub_training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
sub_user_index = sub_training_data[['user_id', 'cate', 'shop_id']].copy()
test = sub_training_data[feats].copy()
print('test shape: ', test.shape)

lgb_train_F12_7(train, label, test, sub_user_index)

"""
F11_7
"""
from user_cate import make_train_set_F11_7, make_test_set_F11_7, lgb_train_F11_7

training_data = make_train_set_F11_7(train_start_date, train_end_date, label_start_date, label_end_date)
sub_training_data = make_test_set_F11_7(test_start_date, test_end_date)

# train
feats_train = [f for f in training_data.columns if f not in ignore_feat]
print(len(feats_train))
label = training_data['label'].copy()
print('train shape: ', training_data.shape)
train = training_data[feats_train].copy()
print('train shape: ', train.shape)

# test
feats_test = [f for f in sub_training_data.columns if f not in ignore_feat]
print(feats_test)
print(len(feats_test))
sub_user_index = sub_training_data[['user_id', 'cate']].copy()
test = sub_training_data[feats_test].copy()
print('test shape: ', test.shape)

# 训练
lgb_train_F11_7(train, label, test, sub_user_index)

"""
F12_5
"""
from  user_cate_shop2 import make_train_set_F12_5, make_test_set_F12_5, lgb_train_F12_5
test_start_date = '2018-03-15'
test_end_date = '2018-04-16'

label_start_date = '2018-04-11'
label_end_date = '2018-04-16'
train_start_date = '2018-03-10'
train_end_date = '2018-04-11'

training_data = make_train_set_F12_5(train_start_date, train_end_date, label_start_date, label_end_date, 30)
feats = [f for f in training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
label = training_data['label'].copy()
train = training_data[feats].values

sub_training_data = make_test_set_F12_5(test_start_date, test_end_date, 30)
feats = [f for f in sub_training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
sub_user_index = sub_training_data[['user_id', 'cate', 'shop_id']].copy()
test = sub_training_data[feats].values
print('test shape: ', test.shape)

lgb_train_F12_5(train, label, test, sub_user_index)

"""
F11_5
"""
from user_cate2 import make_train_set_F11_5, make_test_set_F11_5, lgb_train_F11_5
training_data = make_train_set_F11_5(train_start_date, train_end_date, label_start_date, label_end_date, 30)

feats = [f for f in training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
label = training_data['label'].copy()
user_index = training_data[['user_id', 'cate']].copy()
train = training_data[feats].copy()
del training_data

sub_training_data = make_test_set_F11_5(test_start_date, test_end_date, 30)
feats = [f for f in sub_training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
sub_user_index = sub_training_data[['user_id', 'cate']].copy()
test = sub_training_data[feats].copy()
print('test shape: ', test.shape)
del sub_training_data
lgb_train_F11_5(train, label, test, sub_user_index)

"""
MERGE1
"""
import pandas as pd

f11_col = ['user_id','cate']
f12_col = ['user_id','cate','shop_id']

# 新的的F11 F12结果
f11_new_best_prob = pd.read_csv('./res/sub_F11_7.csv')
f12_new_best_prob = pd.read_csv('./res/sub_F12_7.csv')

# 5天的结果
f12_5days_F11_prob = pd.read_csv('./res/sub_F11_5.csv')
f12_5days_F12_prob = pd.read_csv('./res/sub_F12_5.csv')


def get_old_best(f11_best_prob, f12_best_prob):
   '''
   F11 F12结果融合函数
   '''
   f11_col = ['user_id','cate']
   f12_col = ['user_id','cate','shop_id']
   f11_best_prob = f11_best_prob[f11_col][:32000]
   f12_best_prob = f12_best_prob[f12_col][:32000]
   f11 = f11_best_prob.drop_duplicates(f11_col)[f11_col]
   f11_merge = f11.merge(f12_best_prob, on=f11_col,how='inner')
   all_pred = pd.concat([f12_best_prob.head(15000), f11_merge],axis=0,ignore_index=True)
   output_csv = all_pred.drop_duplicates(f12_col,keep='first')
   print('output_csv.shape:', output_csv.shape)
   return output_csv


# 5天F11F12融合
f12_5days_best_prob = get_old_best(f12_5days_F11_prob,f12_5days_F12_prob)

# 7天F11 F12 融合
new_best = get_old_best(f11_new_best_prob,f12_new_best_prob)

# 7天 & 5天融合
fiveDay20k = f12_5days_best_prob[f12_col].head(20000)
fiveDay15k = f12_5days_best_prob[f12_col].head(15000)
fiveDay10k = f12_5days_best_prob[f12_col].head(10000)
fiveDay5k = f12_5days_best_prob[f12_col].head(5000)

print('原结果:', new_best.shape[0])

merge5day_5k = pd.concat([new_best,fiveDay5k],ignore_index=True)
print('融合5天模型5k去重:', merge5day_5k.drop_duplicates().shape[0])


merge5day_10k = pd.concat([new_best,fiveDay10k],ignore_index=True)
print('融合5天模型10k去重:', merge5day_10k.drop_duplicates().shape[0])

merge5day_15k = pd.concat([new_best,fiveDay15k],ignore_index=True)
print('融合5天模型15k去重:',merge5day_15k.drop_duplicates().shape[0])

merge5day_20k = pd.concat([new_best,fiveDay20k],ignore_index=True)
print('融合5天模型20k去重:',merge5day_20k.drop_duplicates().shape[0])

# 添加你的路径
merge5day_5k.drop_duplicates().to_csv('./res/merge5day_5k.csv',index=False)
merge5day_10k.drop_duplicates().to_csv('./res/merge5day_10k.csv',index=False)
merge5day_15k.drop_duplicates().to_csv('./res/merge5day_15k.csv',index=False)
merge5day_20k.drop_duplicates().to_csv('./res/merge5day_20k.csv',index=False)
new_best.drop_duplicates().to_csv('./res/new_best.csv',index=False)


"""
MERGE2
"""
# 新的的F11 F12结果
f11_best_prob = pd.read_csv('./res/sub_F11_7.csv')
f12_best_prob = pd.read_csv('./res/sub_F12_7.csv')

# 5天的结果
f11_5days_prob = pd.read_csv('./res/sub_F11_5.csv')
f12_5days_prob = pd.read_csv('./res/sub_F12_5.csv')

df1 = pd.merge(f11_best_prob, f11_5days_prob, on=f12_col, how='outer')
df1 = df1.fillna(0)
print(df1.shape)

df2 = pd.merge(f12_best_prob, f12_5days_prob, on=f12_col, how='outer')
df2 = df2.fillna(0)
print(df2.shape)

# 6:4
df1['label']=0.6*df1['label_x']+0.4*df1['label_y']
df1.sort_values(by=['label'], ascending=[0],inplace=True)
df1 = df1.head(32000)
df2['label']=0.6*df2['label_x']+0.4*df2['label_y']
df2.sort_values(by=['label'], ascending=[0],inplace=True)
df2 = df2.head(32000)
print('6/4: ')
sub1 = get_old_best(df1, df2)

# 5:5
df1['label']=0.5*df1['label_x']+0.5*df1['label_y']
df1.sort_values(by=['label'], ascending=[0],inplace=True)
df1 = df1.head(32000)
df2['label']=0.5*df2['label_x']+0.5*df2['label_y']
df2.sort_values(by=['label'], ascending=[0],inplace=True)
df2 = df2.head(32000)
print('5/5: ')
sub2 = get_old_best(df1, df2)

# 7:3
df1['label']=0.7*df1['label_x']+0.3*df1['label_y']
df1.sort_values(by=['label'], ascending=[0],inplace=True)
df1 = df1.head(32000)
df2['label']=0.7*df2['label_x']+0.3*df2['label_y']
df2.sort_values(by=['label'], ascending=[0],inplace=True)
df2 = df2.head(32000)
print('7/3: ')
sub3 = get_old_best(df1, df2)

a = pd.merge(sub1, sub2, on=['user_id', 'cate', 'shop_id'])
b = pd.merge(sub1, sub3, on=['user_id', 'cate', 'shop_id'])
c = pd.merge(sub2, sub3, on=['user_id', 'cate', 'shop_id'])
print("6/4与5/5： ", a.shape)
print("6/4与7/3： ", b.shape)
print("5/5与7/3： ", c.shape)

sub2.drop_duplicates().to_csv('./res/merge_2.csv',index=False)