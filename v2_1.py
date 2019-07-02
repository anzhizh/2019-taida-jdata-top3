# 一个月用户集 全量特征集 一周标签集

from user_cate_shop import *     # 清洗数据

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
user_index = training_data[['user_id', 'cate', 'shop_id']].copy()
train = training_data[feats].copy()

# test
feats = [f for f in sub_training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
sub_user_index = sub_training_data[['user_id', 'cate', 'shop_id']].copy()
test = sub_training_data[feats].copy()
print('test shape: ', test.shape)

del training_data, sub_training_data

lgb_train_F12_7(train, label, test, sub_user_index)


