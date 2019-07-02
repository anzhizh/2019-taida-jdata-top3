# 5天标签
from user_cate2 import *
ignore_feat = ['label', 'type', 'user_id', 'cate','shop_id', 'sku_id','action_time', 'dt', 'market_time', 'shop_reg_tm',
               'user_reg_tm', 'vender_id', 'module_id', 'cate_feat13_11', 'F11_feat13_11']

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
user_index = training_data[['user_id', 'cate']].copy()
train = training_data[feats].copy()
del training_data
# test
sub_training_data = make_test_set(test_start_date, test_end_date, 30)
feats = [f for f in sub_training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
sub_user_index = sub_training_data[['user_id', 'cate']].copy()
test = sub_training_data[feats].copy()
print('test shape: ', test.shape)
del sub_training_data
lgb_train(train, label, test, sub_user_index)


