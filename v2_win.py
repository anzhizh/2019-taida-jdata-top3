# 5天标签
from user_cate_shop2 import *

ignore_feat = ['label', 'type', 'user_id', 'cate','shop_id', 'sku_id','action_time', 'dt', 'market_time', 'shop_reg_tm',
               'user_reg_tm', 'vender_id', 'module_id']

test_start_date = '2018-03-15'
test_end_date = '2018-04-16'

label_start_date = '2018-04-11'
label_end_date = '2018-04-16'
train_start_date = '2018-03-10'
train_end_date = '2018-04-11'

# train
training_data = make_train_set(train_start_date, train_end_date, label_start_date, label_end_date, 30)

feats = [f for f in training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
label = training_data['label'].copy()
user_index = training_data[['user_id', 'cate', 'shop_id']].copy()
train = training_data[feats].values

# test
sub_training_data = make_test_set(test_start_date, test_end_date, 30)
feats = [f for f in sub_training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
sub_user_index = sub_training_data[['user_id', 'cate', 'shop_id']].copy()
test = sub_training_data[feats].values
print('test shape: ', test.shape)

lgb_train_F12_5(train, label, test, sub_user_index)


