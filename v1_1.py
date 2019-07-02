from user_cate import *

ignore_feat = ['label', 'type', 'user_id', 'cate', 'shop_id', 'sku_id', 'action_time', 'dt', 'market_time', 'shop_reg_tm',
               'user_reg_tm']

label_start_date = '2018-04-09'
label_end_date = '2018-04-16'
train_start_date = '2018-03-08'
train_end_date = '2018-04-09'
test_start_date = '2018-03-15'
test_end_date = '2018-04-16'

# train
training_data = make_train_set(train_start_date, train_end_date, label_start_date, label_end_date)

# test
sub_training_data = make_test_set(test_start_date, test_end_date)


# train
feats_train = [f for f in training_data.columns if f not in ignore_feat]
print(len(feats_train))
label = training_data['label'].copy()
user_index = training_data[['user_id', 'cate']].copy()
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
