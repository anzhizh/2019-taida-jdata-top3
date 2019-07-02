from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from dateutil.parser import parse
import warnings
from sklearn import preprocessing
warnings.filterwarnings('ignore')


# 读取用户数据（全量数据）
def get_basic_user_feat():
    dump_path = './cache/basic_user_F12_7.pkl'
    if os.path.exists(dump_path):
        user = pd.read_pickle(dump_path)
    else:
        user = pd.read_pickle('./cache/origin_user.pkl')
        user_info = user.copy()
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        city_level_df = pd.get_dummies(user["city_level"], prefix="city_level")
        province_df = pd.get_dummies(user["province"], prefix="province")
        user = pd.concat([user[['user_id', 'city', 'county']], age_df, sex_df, user_lv_df, city_level_df, province_df],
                         axis=1)

        city_count_map = user_info.city.value_counts()
        province_count_map = user_info.province.value_counts()

        user_info['province_count_map'] = user_info['province'].map(province_count_map).fillna(-1)
        user_info['city_count_map'] = user_info['province'].map(city_count_map).fillna(-1)   #  city

        now = datetime.today()
        user_info['user_reg_tm'] = pd.to_datetime(user_info['user_reg_tm'])
        user_info['user_duration'] = user_info['user_reg_tm'].fillna(now).apply(lambda x: (now - x).days)
        _ = user_info.pop('user_reg_tm')

        user_info.city_level = user_info.city_level.fillna(4.0)
        user_stat = user_info[['user_id', 'province_count_map', 'city_count_map', 'user_duration']]
        user = user.merge(user_stat, on='user_id', how='left')
        pickle.dump(user, open(dump_path, 'wb'))
    print('user finished')
    return user


# 读取行为数据，与产品数据拼接（起始时间-结束时间的行为数据）
def get_actions_product(start_date, end_date):
    dump_path = './cache/all_action_product_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = pd.read_pickle('./cache/origin_action.pkl')
        product = pd.read_pickle('./cache/origin_product.pkl')
        shop = pd.read_pickle('./cache/origin_shop.pkl')
        actions = actions[(actions.action_time >= start_date) & (actions.action_time < end_date)]
        actions = actions[actions['type'] != 5]  # 训练集时间区间内无type=5， 仅测试集时间区间存在
        actions = actions[actions['sku_id'].isin(product['sku_id'])]  # 行为中sku_id不在product中的
        actions = pd.merge(actions, product, on='sku_id', how='left')
        actions = actions[actions['cate'] != 13]  # cate13的数据没有购买行为
        actions = pd.merge(actions, shop[['shop_id', 'vender_id']], on=['shop_id'], how='left')
        print(actions.shape)
        actions = actions[actions['vender_id'] != 3666]  # 数据没有购买行为
        print(actions.shape)
        actions.to_pickle(dump_path)
    return actions


# 读取行为数据，与产品数据拼接（用于生成购物车特征）
def get_actions_product_cart(start_date, end_date):
    dump_path = './cache/all_action_product_cart_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = pd.read_pickle('./cache/origin_action.pkl')
        product = pd.read_pickle('./cache/origin_product.pkl')
        shop = pd.read_pickle('./cache/origin_shop.pkl')
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        actions = actions[(actions.action_time >= start_date) & (actions.action_time < end_date)]
        actions = actions[actions['sku_id'].isin(product['sku_id'])]  # 行为中sku_id不在product中的
        actions = pd.merge(actions, product, on='sku_id', how='left')
        actions = actions[actions['cate'] != 13]  # cate13的数据没有购买行为
        actions = pd.merge(actions, shop[['shop_id', 'vender_id']], on=['shop_id'], how='left')
        print(actions.shape)
        actions = actions[actions['vender_id'] != 3666]  # 数据没有购买行为
        print(actions.shape)
        actions.to_pickle(dump_path)
    return actions


# 行为比例特征（2.01-4.08） 滑窗
def get_accumulate_user_feat_cart(start_date, end_date):
    dump_path = './cache/user_feat_accumulate_cart_F12_7_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        f11_actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product_cart(start_date, end_date)

        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions[['user_id', 'cate', 'shop_id']], df], axis=1)

        # 索引
        f11_actions = actions[['user_id', 'cate', 'shop_id']].drop_duplicates()

        actions1 = actions.drop(['cate', 'shop_id'], axis=1)
        actions1 = actions1.groupby(['user_id'], as_index=False).sum().add_prefix('user_id_')
        actions1['user_action_1_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
            start_date, end_date)] / actions1['user_id_%s-%s-action_1' % (start_date, end_date)]
        actions1['user_action_4_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
            start_date, end_date)] / actions1['user_id_%s-%s-action_4' % (start_date, end_date)]
        actions1['user_action_3_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
            start_date, end_date)] / actions1['user_id_%s-%s-action_3' % (start_date, end_date)]
        actions1['user_action_5_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
            start_date, end_date)] / actions1['user_id_%s-%s-action_5' % (start_date, end_date)]
        actions1.rename(columns={'user_id_user_id': 'user_id'}, inplace=True)

        actions2 = actions.drop(['user_id', 'shop_id'], axis=1)
        actions2 = actions2.groupby(['cate'], as_index=False).sum().add_prefix('cate_')
        actions2['cate_action_1_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
            start_date, end_date)] / actions2['cate_%s-%s-action_1' % (start_date, end_date)]
        actions2['cate_action_4_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
            start_date, end_date)] / actions2['cate_%s-%s-action_4' % (start_date, end_date)]
        actions2['cate_action_3_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
            start_date, end_date)] / actions2['cate_%s-%s-action_3' % (start_date, end_date)]
        actions2['cate_action_5_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
            start_date, end_date)] / actions2['cate_%s-%s-action_5' % (start_date, end_date)]
        actions2.rename(columns={'cate_cate': 'cate'}, inplace=True)

        actions3 = actions.drop(['user_id', 'cate'], axis=1)
        actions3 = actions3.groupby(['shop_id'], as_index=False).sum().add_prefix('shop_id_')
        actions3['shop_action_1_ratio_%s_%s' % (start_date, end_date)] = actions3['shop_id_%s-%s-action_2' % (
            start_date, end_date)] / actions3['shop_id_%s-%s-action_1' % (start_date, end_date)]
        actions3['shop_action_4_ratio_%s_%s' % (start_date, end_date)] = actions3['shop_id_%s-%s-action_2' % (
            start_date, end_date)] / actions3['shop_id_%s-%s-action_4' % (start_date, end_date)]
        actions3['shop_action_3_ratio_%s_%s' % (start_date, end_date)] = actions3['shop_id_%s-%s-action_2' % (
            start_date, end_date)] / actions3['shop_id_%s-%s-action_3' % (start_date, end_date)]
        actions3['shop_action_5_ratio_%s_%s' % (start_date, end_date)] = actions3['shop_id_%s-%s-action_2' % (
            start_date, end_date)] / actions3['shop_id_%s-%s-action_5' % (start_date, end_date)]
        actions3.rename(columns={'shop_id_shop_id': 'shop_id'}, inplace=True)

        actions4 = actions
        actions4 = actions4.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum().add_prefix(
            'user_cate_shop_id_')
        actions4['user_cate_shop_id_action_1_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                              start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_1' % (
                                                                                              start_date, end_date)]
        actions4['user_cate_shop_id_action_4_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                              start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_4' % (
                                                                                              start_date, end_date)]
        actions4['user_cate_shop_id_action_3_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                              start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_3' % (
                                                                                              start_date, end_date)]
        actions4['user_cate_shop_id_action_5_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                              start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_5' % (
                                                                                              start_date, end_date)]
        actions4.rename(columns={'user_cate_shop_id_user_id': 'user_id', 'user_cate_shop_id_cate': 'cate',
                                 'user_cate_shop_id_shop_id': 'shop_id'}, inplace=True)

        # 拼接
        f11_actions = f11_actions.merge(actions1, on='user_id', how='left')
        f11_actions = f11_actions.merge(actions2, on='cate', how='left')
        f11_actions = f11_actions.merge(actions3, on='shop_id', how='left')
        f11_actions = f11_actions.merge(actions4, on=['user_id', 'cate', 'shop_id'], how='left')

    print('accumulate user cart finished')
    return f11_actions


# 行为比例特征（2.01-4.08） 滑窗
def get_accumulate_user_feat(start_date, end_date):
    dump_path = './cache/user_feat_accumulate_F12_7_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        f11_actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)

        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions[['user_id', 'cate', 'shop_id']], df], axis=1)

        # 索引
        f11_actions = actions[['user_id', 'cate', 'shop_id']].drop_duplicates()

        actions1 = actions.drop(['cate', 'shop_id'], axis=1)
        actions1 = actions1.groupby(['user_id'], as_index=False).sum().add_prefix('user_id_')
        actions1['user_action_1_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
            start_date, end_date)] / actions1['user_id_%s-%s-action_1' % (start_date, end_date)]
        actions1['user_action_4_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
            start_date, end_date)] / actions1['user_id_%s-%s-action_4' % (start_date, end_date)]
        actions1['user_action_3_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
            start_date, end_date)] / actions1['user_id_%s-%s-action_3' % (start_date, end_date)]
        actions1.rename(columns={'user_id_user_id': 'user_id'}, inplace=True)

        actions2 = actions.drop(['user_id', 'shop_id'], axis=1)
        actions2 = actions2.groupby(['cate'], as_index=False).sum().add_prefix('cate_')
        actions2['cate_action_1_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
            start_date, end_date)] / actions2['cate_%s-%s-action_1' % (start_date, end_date)]
        actions2['cate_action_4_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
            start_date, end_date)] / actions2['cate_%s-%s-action_4' % (start_date, end_date)]
        actions2['cate_action_3_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
            start_date, end_date)] / actions2['cate_%s-%s-action_3' % (start_date, end_date)]
        actions2.rename(columns={'cate_cate': 'cate'}, inplace=True)

        actions3 = actions.drop(['user_id', 'cate'], axis=1)
        actions3 = actions3.groupby(['shop_id'], as_index=False).sum().add_prefix('shop_id_')
        actions3['shop_action_1_ratio_%s_%s' % (start_date, end_date)] = actions3['shop_id_%s-%s-action_2' % (
            start_date, end_date)] / actions3['shop_id_%s-%s-action_1' % (start_date, end_date)]
        actions3['shop_action_4_ratio_%s_%s' % (start_date, end_date)] = actions3['shop_id_%s-%s-action_2' % (
            start_date, end_date)] / actions3['shop_id_%s-%s-action_4' % (start_date, end_date)]
        actions3['shop_action_3_ratio_%s_%s' % (start_date, end_date)] = actions3['shop_id_%s-%s-action_2' % (
            start_date, end_date)] / actions3['shop_id_%s-%s-action_3' % (start_date, end_date)]
        actions3.rename(columns={'shop_id_shop_id': 'shop_id'}, inplace=True)

        actions4 = actions
        actions4 = actions4.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum().add_prefix(
            'user_cate_shop_id_')
        actions4['user_cate_shop_id_action_1_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                              start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_1' % (
                                                                                              start_date, end_date)]
        actions4['user_cate_shop_id_action_4_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                              start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_4' % (
                                                                                              start_date, end_date)]
        actions4['user_cate_shop_id_action_3_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                              start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_3' % (
                                                                                              start_date, end_date)]
        actions4.rename(columns={'user_cate_shop_id_user_id': 'user_id', 'user_cate_shop_id_cate': 'cate',
                                 'user_cate_shop_id_shop_id': 'shop_id'}, inplace=True)

        # 拼接
        f11_actions = f11_actions.merge(actions1, on='user_id', how='left')
        f11_actions = f11_actions.merge(actions2, on='cate', how='left')
        f11_actions = f11_actions.merge(actions3, on='shop_id', how='left')
        f11_actions = f11_actions.merge(actions4, on=['user_id', 'cate', 'shop_id'], how='left')

    print('accumulate user finished')
    return f11_actions


# 基础统计特征
def get_stat_feat(start_date, end_date):
    dump_path = './cache/stat_feat_accumulate_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        action = pd.read_pickle(dump_path)
    else:
        action = get_actions_product(start_date, end_date)
        action_index = action[['user_id', 'cate', 'shop_id']].drop_duplicates()

        # 行为onehot
        action_type = pd.get_dummies(action['type'])
        action_type.columns = ['act_1', 'act_2', 'act_3', 'act_4']
        action_type = action_type[['act_1', 'act_2', 'act_3', 'act_4']]
        action_type['cate'] = action['cate']
        action_type['user_id'] = action['user_id']
        action_type['shop_id'] = action['shop_id']

        # 基于user_id的统计特征
        user_stat = action[['user_id']].drop_duplicates()
        user_action_count = action.groupby('user_id')['type'].count()
        user_order_count = action_type.groupby('user_id')['act_2'].sum()
        user_order_rate = user_order_count / (user_action_count).fillna(0)
        user_cate_count = action.groupby('user_id')['cate'].nunique()
        user_sku_count = action.groupby('user_id')['sku_id'].nunique()
        user_shop_count = action.groupby('user_id')['shop_id'].nunique()

        user_stat['user_action_count_%s_%s' % (start_date, end_date)] = user_action_count
        user_stat['user_order_rate_%s_%s' % (start_date, end_date)] = user_order_rate
        user_stat['user_cate_count_%s_%s' % (start_date, end_date)] = user_cate_count
        user_stat['user_sku_count_%s_%s' % (start_date, end_date)] = user_sku_count
        user_stat['user_shop_count_%s_%s' % (start_date, end_date)] = user_shop_count

        # 基于cate的统计特征
        cate_stat = action[['cate']].drop_duplicates()

        # cate下的用户特征
        cate_user_count = action.groupby('cate')['user_id'].count()
        cate_user_nunique = action.groupby('cate')['user_id'].nunique()
        cate_order_count = action_type.groupby('cate')['act_2'].sum()
        cate_order_rate = cate_order_count / cate_user_count

        # cate下：购买用户/总用户
        cate_order_user_count = action_type.groupby(['cate', 'user_id'])['act_2'].sum().reset_index()
        cate_order_user_count = cate_order_user_count[cate_order_user_count.act_2 > 0].groupby('cate')[
            'user_id'].nunique()
        cate_order_user_rate = (cate_order_user_count / cate_user_nunique)
        cate_sku_nunique = action.groupby('cate')['sku_id'].nunique()

        # cate下的店铺特征
        cate_shop_count = action.groupby('cate')['shop_id'].count()
        cate_shop_nunique = action.groupby('cate')['shop_id'].nunique()
        cate_shop_order_count = action_type.groupby('cate')['act_2'].sum()
        cate_shop_order_rate = cate_shop_order_count / cate_shop_count

        # cate下： 购买店铺/总店铺
        cate_order_shop_count = action_type.groupby(['cate', 'shop_id'])['act_2'].sum().reset_index()
        cate_order_shop_count = cate_order_shop_count[cate_order_shop_count.act_2 > 0].groupby('cate')[
            'shop_id'].nunique()
        cate_order_shop_rate = (cate_order_shop_count / cate_shop_nunique)

        cate_stat['cate_user_count_%s_%s' % (start_date, end_date)] = cate_user_count
        cate_stat['cate_user_nunique_%s_%s' % (start_date, end_date)] = cate_user_nunique
        cate_stat['cate_order_rate_%s_%s' % (start_date, end_date)] = cate_order_rate.fillna(0)
        cate_stat['cate_order_user_count_%s_%s' % (start_date, end_date)] = cate_order_user_count
        cate_stat['cate_order_user_rate_%s_%s' % (start_date, end_date)] = cate_order_user_rate
        cate_stat['cate_sku_nunique_%s_%s' % (start_date, end_date)] = cate_sku_nunique
        cate_stat['cate_shop_nunique_%s_%s' % (start_date, end_date)] = cate_shop_nunique

        cate_stat['cate_shop_order_rate_%s_%s' % (start_date, end_date)] = cate_shop_order_rate
        cate_stat['cate_order_shop_count_%s_%s' % (start_date, end_date)] = cate_order_shop_count
        cate_stat['cate_order_shop_rate_%s_%s' % (start_date, end_date)] = cate_order_shop_rate

        # 基于shop_id的统计特征
        shop_stat = action[['shop_id']].drop_duplicates()
        shop_user_count = action.groupby('shop_id')['user_id'].count()
        shop_user_nunique = action_type.groupby('shop_id')['user_id'].nunique()
        shop_order_count = action_type.groupby('shop_id')['act_2'].sum()
        shop_order_rate = shop_order_count / (shop_user_count).fillna(0)
        shop_sku_nunique = action.groupby('shop_id')['sku_id'].nunique()
        shop_sku_count = action.groupby('shop_id')['sku_id'].count()

        # 店铺下：购买用户/总用户
        shop_order_user_count = action_type.groupby(['shop_id', 'user_id'])['act_2'].sum().reset_index()
        shop_order_user_count = shop_order_user_count[shop_order_user_count.act_2 > 0].groupby('shop_id')[
            'user_id'].nunique()
        shop_order_user_rate = (shop_order_user_count / shop_user_nunique)

        shop_stat['shop_user_count_%s_%s' % (start_date, end_date)] = shop_user_count
        shop_stat['shop_user_nunique_%s_%s' % (start_date, end_date)] = shop_user_nunique
        shop_stat['shop_order_rate_%s_%s' % (start_date, end_date)] = shop_order_rate
        shop_stat['shop_sku_count_%s_%s' % (start_date, end_date)] = shop_sku_count
        shop_stat['shop_sku_nunique_%s_%s' % (start_date, end_date)] = shop_sku_nunique
        shop_stat['shop_order_user_count_%s_%s' % (start_date, end_date)] = shop_order_user_count
        shop_stat['shop_order_user_rate_%s_%s' % (start_date, end_date)] = shop_order_user_rate

        action = pd.merge(action_index, user_stat, on='user_id', how='left')
        action = pd.merge(action, cate_stat, on='cate', how='left')
        action = pd.merge(action, shop_stat, on='shop_id', how='left')
        action.to_pickle(dump_path)
    print('stat_feat finished')
    return action


def get_hours(start_date, end_date):
    d = parse(end_date) - parse(start_date)
    hours = int(d.days * 24 + d.seconds / 3600)
    return hours


# 行为时间特征
def get_time_feat(start_date, end_date):
    dump_path = './cache/time_feature_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        f11_actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        f11_actions = actions[['user_id']].drop_duplicates()
        active_days = actions[['user_id', 'action_time']]
        buy_days = actions[['user_id', 'action_time', 'type']]
        active_last = actions[['user_id', 'action_time']]
        buy_last = actions[['user_id', 'action_time', 'type']]

        # 活动天数
        active_days = active_days.groupby(['user_id', 'action_time']).size().reset_index()
        active_days = active_days.groupby('user_id').size().reset_index()
        active_days.rename(columns={0: 'user_active_days'}, inplace=True)

        # 购买天数
        buy_days = buy_days[buy_days['type'] == 2]
        del buy_days['type']
        buy_days = buy_days.groupby(['user_id', 'action_time']).size().reset_index()
        buy_days = buy_days.groupby('user_id').size().reset_index()
        buy_days.rename(columns={0: 'user_buy_days'}, inplace=True)

        # 最近交互时间
        active_last = active_last.sort_values(by='action_time', ascending=False)
        active_last = active_last.drop_duplicates('user_id')
        active_last['user_active_last'] = active_last['action_time'].apply(lambda x: get_hours(x, end_date))
        del active_last['action_time']

        # 最近购买时间(h)
        buy_last = buy_last[buy_last['type'] == 2]
        del buy_last['type']
        buy_last = buy_last.sort_values(by='action_time', ascending=False)
        buy_last = buy_last.drop_duplicates('user_id')
        buy_last['user_buy_last'] = buy_last['action_time'].apply(lambda x: get_hours(x, end_date))
        del buy_last['action_time']

        f11_actions = f11_actions.merge(active_days, on='user_id', how='left')
        f11_actions = f11_actions.merge(buy_days, on='user_id', how='left')
        f11_actions = f11_actions.merge(active_last, on='user_id', how='left')
        f11_actions = f11_actions.merge(buy_last, on='user_id', how='left')
        pickle.dump(f11_actions, open(dump_path, 'wb'))
    print('time finished')
    return f11_actions


# 店铺特征
def get_shop_feat(start_date, end_date):
    dump_path = './cache/shop_feature_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        shop_feature = pd.read_pickle(dump_path)
    else:
        shop = pd.read_pickle('./cache/origin_shop.pkl')
        shop = shop[shop['vender_id'] != 3666]
        del shop['vender_id']
        shop = shop.drop_duplicates(['cate', 'shop_id'])
        shop['vip_ratio'] = shop['vip_num'] / shop['fans_num']

        shop_info = shop.copy()
        action_shop = get_actions_product(start_date, end_date)

        now = pd.to_datetime(action_shop.action_time.max())
        shop_info['shop_reg_tm'] = pd.to_datetime(shop_info['shop_reg_tm'])
        shop_info['shop_duration'] = shop_info['shop_reg_tm'].fillna(now).apply(lambda x: (now - x).days)
        _ = shop_info.pop('shop_reg_tm')

        cate_count_map = shop_info.cate.value_counts()
        shop_info['cate_count_map'] = shop_info['cate'].map(cate_count_map).fillna(-1)
        shop_info['cate_shop'] = shop_info['cate'].fillna(-1)
        _ = shop_info.pop('cate')

        action_shop = action_shop[['user_id', 'sku_id', 'action_time', 'shop_id', 'type']]
        action_shop = action_shop.merge(shop_info[['shop_id', 'cate_shop']], on='shop_id', how='left')

        action_shop['order'] = (action_shop.type == 2).astype('int8')
        action_shop['explor'] = (action_shop.type == 1).astype('int8')

        shop_stat = pd.DataFrame()
        shop_stat['shop_order_mean'] = action_shop.groupby('shop_id')['order'].mean()
        shop_stat['shop_order_sum'] = action_shop.groupby('shop_id')['order'].sum()
        shop_stat['shop_act_count'] = action_shop.groupby('shop_id')['order'].count()

        shop_cate_stat = pd.DataFrame()
        shop_cate_stat['shop_cate_order_mean'] = action_shop.groupby('cate_shop')['order'].mean()
        shop_cate_stat['shop_cate_order_sum'] = action_shop.groupby('cate_shop')['order'].sum()
        shop_cate_stat['shop_cate_order_count'] = action_shop.groupby('cate_shop')['order'].count()

        shop_stat = shop_stat.reset_index()
        shop_cate_stat = shop_cate_stat.reset_index()

        shop_info_ = shop_info.merge(shop_stat, on='shop_id', how='left')
        shop_info_ = shop_info_.merge(shop_cate_stat, on='cate_shop', how='left')

        shop_feature = shop_info_.merge(shop_stat, on='shop_id', how='left')
        pickle.dump(shop_feature, open(dump_path, 'wb'))
    del shop_feature['cate_shop']
    print('shop finished')
    return shop_feature


# 商品和评论特征
def get_product_stat_feat(start_date, end_date):
    dump_path = './cache/product_feature_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        cate_stat = pd.read_pickle(dump_path)
    else:
        product_ori = pd.read_pickle('./cache/origin_product.pkl')
        comment_ori = pd.read_pickle('./cache/origin_comment.pkl')
        cate_info = product_ori.copy()
        comment_sku = comment_ori.groupby('sku_id').sum().reset_index()
        cate_info = cate_info.merge(comment_sku, on='sku_id', how='left')

        cate_stat_1 = pd.DataFrame()
        cate_stat_1['cate_sku_count'] = cate_info.groupby('cate')['sku_id'].count()
        cate_stat_1['cate_brand_count'] = cate_info.groupby('cate')['brand'].nunique()
        cate_stat_1['cate_shop_count'] = cate_info.groupby('cate')['shop_id'].nunique()
        cate_stat_1['cate_comments_count'] = cate_info.groupby('cate')['comments'].sum()
        cate_stat_1['cate_good_comments_count'] = cate_info.groupby('cate')['good_comments'].sum()
        cate_stat_1['cate_bad_comments_count'] = cate_info.groupby('cate')['bad_comments'].sum()
        cate_stat_1['cate_good_rate'] = cate_stat_1['cate_good_comments_count'] / cate_stat_1['cate_comments_count']
        cate_stat_1['cate_good_rate'] = cate_stat_1.cate_good_rate.fillna(cate_stat_1.cate_good_rate.mean())
        cate_stat = cate_stat_1.reset_index()
        pickle.dump(cate_stat, open(dump_path, 'wb'))
    print('product finished')
    return cate_stat


def cate_user_reg(d):
    if d <0:
        d = -1
    elif d>=0 and d<=3:
        d = 1
    elif d>3 and d<=6:
        d = 2
    elif d>6 and d<=12:
        d = 3
    elif d>12 and d<=24:
        d = 4
    elif d>24 and d<=48:
        d = 5
    else:
        d = 6
    return d


# 用户特征
def user_features(start_date, end_date):
    dump_path = './cache/user_features_F12_7_%s_2.pkl' % (end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)

    else:
        user = pd.read_pickle('./cache/origin_user.pkl')
        user['reg_duration'] = ((pd.to_datetime(user['user_reg_tm']) - pd.to_datetime('2018/04/16')).dt.days) // 30
        user['reg_duration_cate'] = user['reg_duration'].apply(cate_user_reg)
        sub_action = get_actions_product(start_date, end_date)
        end_date = pd.to_datetime(end_date)
        day = timedelta(1, 0)

        print('=====> 提取特征...')
        sub_action['action_time'] = pd.to_datetime(sub_action['action_time'])
        sub_1 = sub_action[(sub_action['action_time'] >= end_date - 1 * day) & (sub_action['action_time'] < end_date)]
        sub_7 = sub_action[(sub_action['action_time'] >= end_date - 7 * day) & (sub_action['action_time'] < end_date)]
        sub_all = sub_action[sub_action['action_time'] < end_date]

        # ========================================
        #    用户历史行为
        # ========================================
        # 6种行为特征
        df = pd.get_dummies(sub_all['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_all[['user_id', 'cate', 'shop_id']], df], axis=1)

        # 行为sum
        u_feature_all = df.drop(['cate', 'shop_id'], axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id', 'browse_all', 'buy_all', 'follow_all', 'comment_all', 'action_all']
        u_feature_all.columns = col
        u_feature_all['buy/browse_all'] = u_feature_all['buy_all'] / (u_feature_all['browse_all'] + 0.001) * 100
        u_feature_all['buy/follow_all'] = u_feature_all['buy_all'] / (u_feature_all['follow_all'] + 0.001) * 100
        u_feature_all['buy/comment_all'] = u_feature_all['buy_all'] / (u_feature_all['comment_all'] + 0.001) * 100


        # 活跃天数
        u_days = sub_all[['user_id', 'action_time']]
        u_days = u_days.drop_duplicates()
        u_days = u_days.groupby('user_id').count().reset_index()
        u_days.rename(columns={'date': 'u_days_all'}, inplace=True)
        u_feature_all = pd.merge(u_feature_all, u_days, on='user_id', how='left').fillna(0)

        # 时间特征
        u_days = sub_all[['user_id', 'action_time']]
        u_start = u_days.groupby('user_id').min().reset_index()
        u_start.rename(columns={'action_time': 'start'}, inplace=True)
        u_end = u_days.groupby('user_id').max().reset_index()
        u_end.rename(columns={'action_time': 'end'}, inplace=True)
        u_duration = pd.merge(u_start, u_end, on='user_id')
        u_duration['u_duration_all'] = u_duration['end'] - u_duration['start']
        u_duration['u_duration_all'] = u_duration['u_duration_all'].map(lambda x: x.days * 24 + x.seconds / 3600)
        u_duration = u_duration[['user_id', 'u_duration_all']]
        u_feature_all = pd.merge(u_feature_all, u_duration, on='user_id', how='left').fillna(0)

        # 行为/时间
        u_feature_all['action_avg_all'] = u_feature_all['action_all'] / (u_feature_all['u_duration_all'] + 0.001)
        u_feature_all['browse_avg_all'] = u_feature_all['browse_all'] / (u_feature_all['u_duration_all'] + 0.001)
        u_feature_all['buy_avg_all'] = u_feature_all['buy_all'] / (u_feature_all['u_duration_all'] + 0.001)
        u_feature_all['follow_avg_all'] = u_feature_all['follow_all'] / (u_feature_all['u_duration_all'] + 0.001)
        u_feature_all['comment_avg_all'] = u_feature_all['comment_all'] / (u_feature_all['u_duration_all'] + 0.001)

        # ========================================
        #     用户7天行为特征
        # ========================================
        df = pd.get_dummies(sub_7['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_7[['user_id', 'cate', 'shop_id']], df], axis=1)

        # 子集行为特征
        u_feature_7 = df.drop(['cate', 'shop_id'], axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id', 'browse_7', 'buy_7', 'follow_7', 'comment_7', 'action_7']
        u_feature_7.columns = col

        # 时间特征
        u_days = sub_7[['user_id', 'action_time']]
        u_start = u_days.groupby('user_id').min().reset_index()
        u_start.rename(columns={'action_time': 'start'}, inplace=True)
        u_end = u_days.groupby('user_id').max().reset_index()
        u_end.rename(columns={'action_time': 'end'}, inplace=True)
        u_duration = pd.merge(u_start, u_end, on='user_id')
        u_duration['u_duration_7'] = u_duration['end'] - u_duration['start']
        u_duration['u_duration_7'] = u_duration['u_duration_7'].map(lambda x: x.days * 24 + x.seconds / 3600)
        u_duration['u_stop_7'] = end_date - u_duration['end']
        u_duration['u_stop_7'] = u_duration['u_stop_7'].map(lambda x: x.days * 24 + x.seconds / 3600)
        u_duration = u_duration[['user_id', 'u_duration_7', 'u_stop_7']]
        u_feature_7 = pd.merge(u_feature_7, u_duration, on='user_id', how='left').fillna(0)

        # ========================================
        #     用户1天行为特征
        # ========================================
        df = pd.get_dummies(sub_1['type'], prefix='type')
        df['type_0'] = df.sum(axis=1)
        df = pd.concat([sub_1[['user_id', 'cate', 'shop_id']], df], axis=1)

        u_feature_1 = df.drop(['cate', 'shop_id'], axis=1).groupby('user_id').sum().reset_index()
        col = ['user_id', 'browse_1', 'buy_1', 'follow_1', 'comment_1', 'action_1']
        u_feature_1.columns = col

        # ========================================
        #          特征融合
        # ========================================
        actions = pd.merge(user[['user_id', 'user_lv_cd', 'reg_duration', 'reg_duration_cate']], u_feature_all,
                           on='user_id', how='left')
        actions['lv/reg_day'] = actions['user_lv_cd'] / (actions['reg_duration'] + 0.001) * 100
        actions['lv/reg_day_cate'] = actions['user_lv_cd'] / (actions['reg_duration_cate'] + 0.001)
        actions = pd.merge(actions, u_feature_7, on='user_id', how='left')
        actions['action_7D/all'] = actions['action_7'] / (actions['action_all'] + 0.001)
        actions = pd.merge(actions, u_feature_1, on='user_id', how='left')
        actions['action_diff1'] = actions['action_1'] - actions['action_avg_all']
        actions['browse_diff1'] = actions['browse_1'] - actions['browse_avg_all']
        actions['buy_diff1'] = actions['buy_1'] - actions['buy_avg_all']
        actions['follow_diff1'] = actions['follow_1'] - actions['follow_avg_all']
        actions['comment_diff1'] = actions['comment_1'] - actions['comment_avg_all']

        actions['action_diff7'] = actions['action_7'] - actions['action_avg_all']
        actions['browse_diff7'] = actions['browse_7'] - actions['browse_avg_all']
        actions['buy_diff7'] = actions['buy_7'] - actions['buy_avg_all']
        actions['follow_diff7'] = actions['follow_7'] - actions['follow_avg_all']
        actions['comment_diff7'] = actions['comment_7'] - actions['comment_avg_all']

        col = ['browse_7', 'buy_7', 'follow_7', 'comment_7', 'action_7','user_lv_cd', 'browse_1', 'buy_1',
               'follow_1', 'comment_1', 'action_1']
        actions = actions.drop(col, axis=1)
        print(actions.columns)
        print('=====> 完成!')
        pickle.dump(actions, open(dump_path, 'wb'))
    print(actions.shape)
    print('user feat finished')
    return actions


# 交叉特征
def get_cross_feat(start_date, end_date):
    dump_path = './cache/cross_feat_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id']]
        actions['cnt'] = 0

        action1 = actions.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()

        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['cate']
        del action2['shop_id']
        action2.columns = ['user_id', 'user_cnt']

        action3 = actions.groupby('cate', as_index=False).count()
        del action3['user_id']
        del action3['shop_id']
        action3.columns = ['cate', 'cate_cnt']

        action4 = actions.groupby('shop_id', as_index=False).count()
        del action4['user_id']
        del action4['cate']
        action4.columns = ['shop_id', 'shop_cnt']

        actions = pd.merge(action1, action2, how='left', on='user_id')
        actions = pd.merge(actions, action3, how='left', on='cate')
        actions = pd.merge(actions, action4, how='left', on='shop_id')

        actions['user_cnt'] = actions['cnt'] / actions['user_cnt']
        actions['cate_cnt'] = actions['cnt'] / actions['cate_cnt']
        actions['shop_cnt'] = actions['cnt'] / actions['shop_cnt']
        del actions['cnt']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate', 'shop_id'] + ['cross_feat_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('cross feature finished')
    return actions


# 加购特征
def get_last1day_cart_fearture(start_date, end_date, day):
    '''
    设计两个特征
    第一个是在f12id上act5的总和
    第二个是f12id act5行为总和 * (act_2==0)
    '''
    this_end_date = pd.to_datetime(end_date)
    this_start_date = this_end_date - timedelta(days=day)

    # date转化为str
    this_end_date = str(this_end_date).split(' ')[0]
    this_start_date = str(this_start_date).split(' ')[0]
    x_action = get_actions_product_cart(this_start_date, this_end_date)
    print('from:', x_action.action_time.min(), '  to:', x_action.action_time.max())

    x_oh = pd.get_dummies(x_action.type, prefix='act').astype('int8')

    x_action_oh = pd.concat([x_action[['user_id', 'cate', 'shop_id', 'sku_id', 'action_time']], x_oh], axis=1)

    x_act5_stat = x_action_oh.groupby(['user_id', 'cate', 'shop_id'])[['act_5', 'act_2']].sum().add_prefix(
        'lastday_sum_').reset_index()

    x_act5_stat['cart_not_buy'] = x_act5_stat['lastday_sum_act_5'] * (x_act5_stat['lastday_sum_act_2'] == 0)

    #x_act5_stat['cart_minus_buy'] = x_act5_stat['lastday_sum_act_5'] - x_act5_stat['lastday_sum_act_2']

    return x_act5_stat
"""
########################
user_id feat
########################
"""


# 行为前的累积特征(访问天数)
def get_user_feat1(start_date, end_date):
    dump_path = './cache/user_feat1_after_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 购买前的浏览天数
        def user_feat_2_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(visit, buy, on='user_id', how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 用户关注前访问天数
        def user_feat_3_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='user_id', how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注天数
        def user_feat_2_5(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.drop_duplicates(['user_id', 'action_time'], keep='first')
            del guanzhu['action_time']
            del actions['action_time']
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='user_id', how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(user_feat_2_1(start_date, end_date), user_feat_3_1(start_date, end_date), on='user_id',
                           how='outer')
        actions = pd.merge(actions, user_feat_2_5(start_date, end_date), on='user_id', how='outer')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat1_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat1 finished')
    return actions


# 用户平均访问间隔 慢
def get_user_feat2(start_date, end_date):
    dump_path = './cache/user_feat2_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'action_time']]
        df['action_time'] = df['action_time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['user_id', 'action_time'], keep='first')
        df['action_time'] = df['action_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby('user_id', as_index=False).agg(lambda x: x['action_time'].diff().mean())
        actions['avg_visit'] = actions['action_time'].dt.days
        del actions['action_time']
        pickle.dump(actions, open(dump_path, 'wb'))
    print('get_user_feat2 finished')
    return actions


# 用户平均4种行为的访问间隔 慢
def get_user_feat3(start_date, end_date):
    dump_path = './cache/user_feat3_six_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'action_time', 'type']]
        df = df.dropna()
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(start_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] * (-1)
        df = df.drop_duplicates(['user_id', 'action_time', 'type'], keep='first')
        actions = df.groupby(['user_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat3_six_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat3 finished')
    return actions


# 用户的购买频率  慢
def get_user_feat4(start_date, end_date):
    dump_path = './cache/user_feat4_six_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'type', 'action_time']]
        df['action_time'] = pd.to_datetime(df['action_time'])
        actions = df.groupby(['user_id', 'type'], as_index=False).count()
        time_min = df.groupby(['user_id', 'type'], as_index=False).min()
        time_max = df.groupby(['user_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['user_id', 'type'], how='left')
        time_cha['action_time_x'] = pd.to_datetime(time_cha['action_time_x'])
        time_cha['action_time_y'] = pd.to_datetime(time_cha['action_time_y'])
        time_cha['cha_hour'] = 1 + (time_cha['action_time_x'] - time_cha['action_time_y']).dt.days * 24 + \
                               (time_cha['action_time_x'] - time_cha['action_time_y']).dt.seconds // 3600
        del time_cha['action_time_x']
        del time_cha['action_time_y']

        actions = pd.merge(time_cha, actions, on=['user_id', 'type'], how="left")
        actions = actions.groupby(['user_id', 'type']).sum()
        actions['cnt/time'] = actions['action_time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat4_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat4 finished')
    return actions


# 行为0-1化
def user_top_k_0_1(start_date, end_date):
    actions = get_actions_product(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby('user_id', as_index=False).sum()
    del actions['type']
    del actions['sku_id']
    user_id = actions['user_id']
    del actions['user_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([user_id, actions], axis=1)
    return actions


# 用户最近K天行为0/1提取
def get_user_feat5(start_date, end_date):
    dump_path = './cache/user_feat5_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = user_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, user_top_k_0_1(start_days, end_date), how='outer', on='user_id')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat5_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat5 finished')
    return actions


# 获取用户的重复购买率
def get_user_feat6(start_date, end_date):
    dump_path = './cache/product_feat6_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'sku_id', 'type']]
        df = df[df['type'] == 2]  # 购买的行为
        df = df.groupby(['user_id', 'sku_id'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['user_id'], as_index=False)
        actions = grouped.count()[['user_id', 'count1']]
        actions.columns = ['user_id', 'count']
        re_count = grouped.sum()[['user_id', 'count1']]
        re_count.columns = ['user_id', 're_count']
        actions = pd.merge(actions, re_count, on='user_id', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['user_id'], re_buy_rate], axis=1)
        actions.columns = ['user_id', 're_buy_rate']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat6_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat6 finished')
    return actions


# 获取最近一次行为的时间距离当前时间的差距
def get_user_feat7(start_date, end_date):
    dump_path = './cache/user_feat7_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'action_time', 'type']]
        df = df.drop_duplicates(['user_id', 'type'], keep='last')
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(end_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] + 1
        actions = df.groupby(['user_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat7_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat7 finished')
    return actions


# 用户购买/加入购物车/关注前访问次数
def get_user_feat8(start_date, end_date):
    dump_path = './cache/user_feat8_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 用户购买前访问次数
        def user_feat_8_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(visit, buy, on='user_id', how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 用户关注前访问次数
        def user_feat_8_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='user_id', how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注次数
        def user_feat_8_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'type']]
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='user_id', how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(user_feat_8_1(start_date, end_date), user_feat_8_2(start_date, end_date), on='user_id',
                           how='outer')
        actions = pd.merge(actions, user_feat_8_3(start_date, end_date), on='user_id', how='outer')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat8_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat8 finished')
    return actions


# 用户行为的交叉
def get_user_feat9(start_date, end_date):
    dump_path = './cache/user_feat16_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)[['user_id', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['user_id', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['type']
        action2.columns = ['user_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='user_id')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat9_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat9 finished')
    return actions


# 获取最后一次行为的次数
def get_user_feat10(start_date, end_date):
    dump_path = './cache/user_feat10_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:

        df = get_actions_product(start_date, end_date)[['user_id', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(end_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] + 1
        idx = df.groupby(['user_id', 'type'])['action_time'].transform(min)
        idx1 = idx == df['action_time']
        actions = df[idx1].groupby(["user_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user = actions[['user_id']]
        del actions['user_id']
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat10_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat10 finished')
    return actions


# 用户浏览 关注到购买的时间间隔
def get_user_feat11(start_date, end_date):
    dump_path = './cache/get_user_feat11_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        actions_liulan = actions[actions['type'] == 1][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_liulan['time_liulan'] = actions_liulan[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_liulan = actions_liulan[['user_id', 'cate', 'shop_id', 'time_liulan']]
        actions_liulan = actions_liulan.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_guanzhu = actions[actions['type'] == 3][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_guanzhu['time_guanzhu'] = actions_guanzhu[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_guanzhu = actions_guanzhu[['user_id', 'cate', 'shop_id', 'time_guanzhu']]
        actions_guanzhu = actions_guanzhu.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_goumai = actions[actions['type'] == 2][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_goumai['time_goumai'] = actions_goumai[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'cate', 'shop_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='last')

        actions1 = pd.merge(actions_liulan, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions1['time_jiange'] = actions1['time_goumai'] - actions1['time_liulan']
        actions1 = actions1.drop(['cate', 'shop_id', 'time_goumai', 'time_liulan'], axis=1)
        actions1['time_jiange'] = actions1['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min1 = actions1.groupby('user_id').min().reset_index()
        actions_min1.columns = ['user_id', 'time_min1']
        actions_max1 = actions1.groupby('user_id').max().reset_index()
        actions_max1.columns = ['user_id', 'time_max1']
        del actions1

        actions2 = pd.merge(actions_guanzhu, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions2['time_jiange'] = actions2['time_goumai'] - actions2['time_guanzhu']
        actions2 = actions2.drop(['cate', 'shop_id', 'time_goumai', 'time_guanzhu'], axis=1)
        actions2['time_jiange'] = actions2['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min2 = actions2.groupby('user_id').min().reset_index()
        actions_min2.columns = ['user_id', 'time_min2']
        actions_max2 = actions2.groupby('user_id').max().reset_index()
        actions_max2.columns = ['user_id', 'time_max2']
        del actions2

        actions = pd.merge(actions_min1, actions_max1, on='user_id', how='left')
        actions = pd.merge(actions, actions_min2, on='user_id', how='left')
        actions = pd.merge(actions, actions_max2, on='user_id', how='left')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat11_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat11 finished')
    return actions


# 用户购买每种cate的数量
def get_user_feat12(start_date, end_date):
    dump_path = './cache/get_user_feat12_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions = get_actions_product(start_date, end_date)[['user_id', 'cate']]
        cate_col = pd.get_dummies(actions['cate'], prefix='cate')
        actions = pd.concat([actions[['user_id']], cate_col], axis=1)
        actions = actions.groupby('user_id').sum().reset_index()

        user_id = actions[['user_id']]
        del actions['user_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions, columns=columns)], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    print('get_user_feat12 finished')
    return actions


# 用户总购买/关注/浏览/评论品牌数
def get_user_feat13(start_date, end_date):
    dump_path = './cache/user_feat13_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        action = None
        for i in (1, 2, 3, 4):
            df = actions[actions['type'] == i][['user_id', 'cate', 'shop_id']]
            df = df.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')
            df = df.groupby('user_id', as_index=False).count()
            df.columns = ['user_id', 'num_%s' % i, 'num1_%s' % i]
            if i == 1:
                action = df
            else:
                action = pd.merge(action, df, on='user_id', how='outer')
        actions = action.fillna(0)
        actions = actions.astype('float')
        user = actions[['user_id']]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.drop(['user_id'], axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['u_feat13_' + str(i) for i in range(1, actions.shape[1])]
    print('get_user_feat13 finished')
    return actions


# 最早 最晚交互时间 总活跃时间
def get_user_feat14(start_date,end_date):
    dump_path = './cache/user_feat14_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions=get_actions_product(start_date,end_date)[['user_id','action_time']]
        actions1=actions.groupby('user_id',as_index=False).first()
        actions1['time_diff_early']=pd.to_datetime(end_date)-pd.to_datetime(actions1['action_time'])
        actions1['time_diff_early']=actions1['time_diff_early'].dt.days*24+actions1['time_diff_early'].dt.seconds//3600
        actions1=actions1[['user_id','time_diff_early']]

        actions2 = actions.groupby('user_id', as_index=False).last()
        actions2['time_diff_recent'] = pd.to_datetime(end_date) - pd.to_datetime(actions2['action_time'])
        actions2['time_diff_recent'] = actions2['time_diff_recent'].dt.days * 24 + actions2[
            'time_diff_recent'].dt.seconds // 3600
        actions2 = actions2[['user_id', 'time_diff_recent']]

        actions['action_time'] = pd.to_datetime(actions['action_time']).dt.date
        actions = actions.drop_duplicates(['user_id', 'action_time'])[['user_id', 'action_time']]
        actions = actions.groupby('user_id', as_index=False).count()

        actions = pd.merge(actions, actions1,  on='user_id', how='left')
        actions = pd.merge(actions, actions2, on='user_id', how='left')
        actions=actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        action = min_max_scale.fit_transform(actions.drop(['user_id'], axis=1).values)
        actions = pd.concat([actions[['user_id']], pd.DataFrame(action)], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns=['user_id']+['u_feat14_'+str(i)for i in range(1,actions.shape[1])]
    print('get_user_feat14 finished')
    return actions


# U_B对行为1，2，4，5进行 浏览次数/用户总浏览次数（或者物品的浏览次数）
def get_user_feat15(start_date, end_date):
    dump_path = './cache/user_feat15_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
        actions.columns = ['user_id', 'cate', 'shop_id'] + ['user_feat15_' + str(i) for i in
                                                            range(1, actions.shape[1] - 2)]
        return actions
    else:
        temp = None
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type']]
        for i in (1, 2, 3):
            actions = df[df['type'] == i]
            action1 = actions.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            action1.columns = ['user_id', 'cate', 'shop_id', 'visit']

            action2 = actions.groupby('user_id', as_index=False).count()
            del action2['type']
            del action2['shop_id']
            action2.columns = ['user_id', 'user_visits_cate']

            action3 = actions.groupby('user_id', as_index=False).count()
            del action3['type']
            del action3['cate']
            action3.columns = ['user_id', 'user_visits_shop']

            action4 = actions.groupby('cate', as_index=False).count()
            del action4['type']
            action4.columns = ['cate', 'cate_visits_user', 'cate_visits_shop']

            action5 = actions.groupby('shop_id', as_index=False).count()
            del action5['type']
            del action5['cate']
            action5.columns = ['shop_id', 'shop_visits_user']

            action6 = actions.groupby(['cate', 'shop_id'], as_index=False).count()           #######################################
            del action6['type']
            action6.columns = ['cate','shop_id', 'F11_visits_user']

            action7 = actions.groupby(['user_id', 'cate'],
                                      as_index=False).count()  #######################################
            del action7['type']
            action7.columns = ['user_id', 'cate', 'F11_visits_shop']

            actions = pd.merge(action1, action2, how='left', on='user_id')
            actions = pd.merge(actions, action3, how='left', on='user_id')
            actions = pd.merge(actions, action4, how='left', on='cate')
            actions = pd.merge(actions, action5, how='left', on='shop_id')
            actions = pd.merge(actions, action6, how='left', on=['cate', 'shop_id'])
            actions = pd.merge(actions, action7, how='left', on=['user_id', 'cate'])

            actions['visit_rate_user1'] = actions['visit'] / actions['user_visits_cate']
            actions['visit_rate_user2'] = actions['visit'] / actions['user_visits_shop']

            actions['visit_rate_cate1'] = actions['visit'] / actions['cate_visits_user']
            actions['visit_rate_cate2'] = actions['visit'] / actions['cate_visits_shop']

            actions['visit_rate_shop'] = actions['visit'] / actions['shop_visits_user']

            actions['visit_rate_F11'] = actions['visit'] / actions['F11_visits_user']
            actions['visit_rate_F11_1'] = actions['visit'] / actions['F11_visits_shop']

            cols = ['visit', 'user_visits_cate', 'user_visits_shop', 'cate_visits_user', 'cate_visits_shop',
                    'shop_visits_user', 'F11_visits_user', 'F11_visits_shop']
            actions = actions.drop(cols, axis=1)
            if temp is None:
                temp = actions
            else:
                temp = pd.merge(temp, actions, how="outer", on=['user_id', 'cate', 'shop_id'])
        pickle.dump(temp, open(dump_path, 'wb'))
        temp.columns = ['user_id', 'cate', 'shop_id'] + ['user_feat15_' + str(i) for i in
                                                            range(1, temp.shape[1] - 2)]
        return temp

"""
#########################
cate类特征
#########################
"""


# cate购买/加入购物车/关注前访问天数
def get_cate_feat_1(start_date, end_date):
    dump_path = './cache/cate_feat1_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # cate购买前访问天数
        def cate_feat_1_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['cate', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['cate', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby('cate', as_index=False).count()
            visit.columns = ['cate', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('cate', as_index=False).count()
            buy.columns = ['cate', 'buy']
            actions = pd.merge(visit, buy, on='cate', how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问天数
        def cate_feat_1_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['cate', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['cate', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby('cate', as_index=False).count()
            visit.columns = ['cate', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('cate', as_index=False).count()
            guanzhu.columns = ['cate', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='cate', how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注天数
        def cate_feat_1_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['cate', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.drop_duplicates(['cate', 'action_time'], keep='first')
            del guanzhu['action_time']
            del actions['action_time']
            guanzhu = guanzhu.groupby('cate', as_index=False).count()
            guanzhu.columns = ['cate', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('cate', as_index=False).count()
            buy.columns = ['cate', 'buy']
            actions = pd.merge(guanzhu, buy, on='cate', how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(cate_feat_1_1(start_date, end_date), cate_feat_1_2(start_date, end_date),
                           on='cate', how='outer')
        actions = pd.merge(actions, cate_feat_1_3(start_date, end_date), on='cate', how='outer')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat1_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat1 finished')
    return actions


# cate平均访问间隔
def get_cate_feat_2(start_date, end_date):
    dump_path = './cache/cate_feat2_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['cate', 'action_time']]
        df['action_time'] = df['action_time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['cate', 'action_time'], keep='first')
        df['action_time'] = df['action_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby('cate', as_index=False).agg(lambda x: x['action_time'].diff().mean())
        actions['avg_visit'] = actions['action_time'].dt.days
        del actions['action_time']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat2_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat2 finished')
    return actions


# cat四种行为平均访问间隔
def get_cate_feat_3(start_date, end_date):
    dump_path = './cache/cate_feat3_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['cate', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(start_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] * (-1)
        df = df.drop_duplicates(['cate', 'action_time', 'type'], keep='first')
        actions = df.groupby(['cate', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat3_six_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat3 finished')
    return actions


# 最近K天
def product_top_k_0_1(start_date, end_date):
    actions = get_actions_product(start_date, end_date)
    actions = actions[['user_id', 'cate', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby('cate', as_index=False).sum()
    del actions['type']
    del actions['user_id']
    cate = actions['cate']
    del actions['cate']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([cate, actions], axis=1)
    return actions


# 最近K天行为0/1提取
def get_cate_feat_4(start_date, end_date):
    dump_path = './cache/cate_feat4_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = product_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, product_top_k_0_1(start_days, end_date), how='outer', on='cate')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat4_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat4 finished')
    return actions


# 商品的重复购买率
def get_cate_feat_5(start_date, end_date):
    dump_path = './cache/cate_feat5_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type']]
        df = df[df['type'] == 2]  # 购买的行为
        df = df.groupby(['user_id', 'cate'], as_index=False).count()
        df.columns = ['user_id', 'cate', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['cate'], as_index=False)
        actions = grouped.count()[['cate', 'count1']]
        actions.columns = ['cate', 'count']
        re_count = grouped.sum()[['cate', 'count1']]
        re_count.columns = ['cate', 're_count']
        actions = pd.merge(actions, re_count, on='cate', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['cate'], re_buy_rate], axis=1)
        actions.columns = ['cate', 're_buy_rate']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat5_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat5 finished')
    return actions


# 获取货物最近一次行为的时间距离当前时间的差距
def get_cate_feat_6(start_date, end_date):
    dump_path = './cache/cate_feat6_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['cate', 'action_time', 'type']]
        df = df.drop_duplicates(['cate', 'type'], keep='last')
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(end_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] + 1
        actions = df.groupby(['cate', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat6_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat6 finished')
    return actions


# 商品购买/加入购物车/关注前访问次数
def get_cate_feat_7(start_date, end_date):
    dump_path = './cache/cate_feat7_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 商品购买前访问次数
        def cate_feat_7_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['cate', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('cate', as_index=False).count()
            visit.columns = ['cate', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('cate', as_index=False).count()
            buy.columns = ['cate', 'buy']
            actions = pd.merge(visit, buy, on='cate', how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问次数
        def cate_feat_7_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['cate', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('cate', as_index=False).count()
            visit.columns = ['cate', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('cate', as_index=False).count()
            guanzhu.columns = ['cate', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='cate', how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注次数
        def cate_feat_7_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['cate', 'type']]
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('cate', as_index=False).count()
            guanzhu.columns = ['cate', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('cate', as_index=False).count()
            buy.columns = ['cate', 'buy']
            actions = pd.merge(guanzhu, buy, on='cate', how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(cate_feat_7_1(start_date, end_date), cate_feat_7_2(start_date, end_date), on='cate',
                           how='outer')
        actions = pd.merge(actions, cate_feat_7_3(start_date, end_date), on='cate', how='outer')
        cate = actions['cate']
        del actions['cate']
        actions = actions.fillna(0)
        actions = pd.concat([cate, pd.DataFrame(actions)], axis=1)

        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat7_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat7 finished')
    return actions


# 商品行为的交叉
def get_cate_feat_8(start_date, end_date):
    dump_path = './cache/cate_feat8_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)[['cate', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['cate', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby('cate', as_index=False).count()
        del action2['type']
        action2.columns = ['cate', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='cate')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat8_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat8 finished')
    return actions


# 获取最后一次行为的次数
def get_cate_feat_9(start_date, end_date):
    dump_path = './cache/cate_feat9_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:

        df = get_actions_product(start_date, end_date)[['cate', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(end_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] + 1
        idx = df.groupby(['cate', 'type'])['action_time'].transform(min)
        idx1 = idx == df['action_time']
        actions = df[idx1].groupby(['cate', "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user = actions[['cate']]
        del actions['cate']
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat9_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat9 finished')
    return actions


# 用户浏览 关注到购买的时间间隔
def get_cate_feat_10(start_date, end_date):
    dump_path = './cache/cate_feat10_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        actions_liulan = actions[actions['type'] == 1][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_liulan['time_liulan'] = actions_liulan[
            'action_time']
        actions_liulan = actions_liulan[['user_id', 'cate', 'shop_id', 'time_liulan']]
        actions_liulan = actions_liulan.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_guanzhu = actions[actions['type'] == 3][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_guanzhu['time_guanzhu'] = actions_guanzhu[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_guanzhu = actions_guanzhu[['user_id', 'cate', 'shop_id', 'time_guanzhu']]
        actions_guanzhu = actions_guanzhu.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_goumai = actions[actions['type'] == 2][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_goumai['time_goumai'] = actions_goumai[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'cate', 'shop_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='last')

        actions1 = pd.merge(actions_liulan, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions1['time_jiange'] = actions1['time_goumai'] - actions1['time_liulan']
        actions1 = actions1.drop(['user_id', 'shop_id', 'time_goumai', 'time_liulan'], axis=1)
        actions1['time_jiange'] = actions1['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min1 = actions1.groupby('cate').min().reset_index()
        actions_min1.columns = ['cate', 'time_min1']
        actions_max1 = actions1.groupby('cate').max().reset_index()
        actions_max1.columns = ['cate', 'time_max1']
        del actions1

        actions2 = pd.merge(actions_guanzhu, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions2['time_jiange'] = actions2['time_goumai'] - actions2['time_guanzhu']
        actions2 = actions2.drop(['user_id', 'shop_id', 'time_goumai', 'time_guanzhu'], axis=1)
        actions2['time_jiange'] = actions2['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min2 = actions2.groupby('cate').min().reset_index()
        actions_min2.columns = ['cate', 'time_min2']
        actions_max2 = actions2.groupby('cate').max().reset_index()
        actions_max2.columns = ['cate', 'time_max2']
        del actions2

        actions = pd.merge(actions_min1, actions_max1, on='cate', how='left')
        actions = pd.merge(actions, actions_min2, on='cate', how='left')
        actions = pd.merge(actions, actions_max2, on='cate', how='left')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat10_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat10 finished')
    return actions


# 用户总购买/关注/浏览/评论品牌数
def get_cate_feat_11(start_date, end_date):
    dump_path = './cache/cate_feat11_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        action = None
        for i in (1, 2, 3, 4):
            df = actions[actions['type'] == i][['cate', 'shop_id']]
            df = df.drop_duplicates(['cate', 'shop_id'], keep='first')
            df = df.groupby('cate', as_index=False).count()
            df.columns = ['cate', 'num_%s' % i]
            if i == 1:
                action = df
            else:
                action = pd.merge(action, df, on='cate', how='outer')
        actions = action.fillna(0)
        actions = actions.astype('float')
        user = actions[['cate']]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.drop(['cate'], axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat11_' + str(i) for i in range(1, actions.shape[1])]
    print('cate feat11 finished')
    return actions


# 层级的天数
def get_cate_feat_12(start_date, end_date, n):
    dump_path = './cache/cate_feat12_F12_7_%s_%s_%s.pkl' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['cate', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(end_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] // n
        df = df.drop_duplicates(['cate', 'type', 'action_time'], keep='first')
        actions = df.groupby(['cate', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['cate']]
        del actions['cate']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat12_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    print('get cate feat12 finished')
    return actions


# 用户每隔7天购买次数
def get_cate_feat_13(start_date, end_date):
    dump_path = './cache/cate_feat13_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        n = 7
        df = get_actions_product(start_date, end_date)[['cate', 'action_time', 'type']]
        df = df[df['type'] == 2][['cate', 'action_time']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df['action_time'] = pd.to_datetime(df['action_time'])
        df['action_time'] = (pd.to_datetime(end_date) - df['action_time']).dt.days
        df['action_time'] = df['action_time'] // n
        days = np.max(df['action_time'])

        df['cnt'] = 0
        actions = df.groupby(['cate', 'action_time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['cate']]
        del actions['cate']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['cate_feat13_' + str(i) for i in range(1, actions.shape[1])]
    print('get cate feat13 finished')
    return actions


"""
#######################
shop_id类特征
#######################
"""


# shop_id购买/加入购物车/关注前访问天数
def get_shop_id_feat_1(start_date, end_date):
    dump_path = './cache/shop_id_feat1_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # shop_id购买前访问天数
        def shop_id_feat_1_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['shop_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['shop_id', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby('shop_id', as_index=False).count()
            visit.columns = ['shop_id', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('shop_id', as_index=False).count()
            buy.columns = ['shop_id', 'buy']
            actions = pd.merge(visit, buy, on='shop_id', how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问天数
        def shop_id_feat_1_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['shop_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['shop_id', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby('shop_id', as_index=False).count()
            visit.columns = ['shop_id', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('shop_id', as_index=False).count()
            guanzhu.columns = ['shop_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='shop_id', how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注天数
        def shop_id_feat_1_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['shop_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.drop_duplicates(['shop_id', 'action_time'], keep='first')
            del guanzhu['action_time']
            del actions['action_time']
            guanzhu = guanzhu.groupby('shop_id', as_index=False).count()
            guanzhu.columns = ['shop_id', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('shop_id', as_index=False).count()
            buy.columns = ['shop_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='shop_id', how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(shop_id_feat_1_1(start_date, end_date), shop_id_feat_1_2(start_date, end_date),
                           on='shop_id', how='outer')
        actions = pd.merge(actions, shop_id_feat_1_3(start_date, end_date), on='shop_id', how='outer')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat1_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat1 finished')
    return actions


# shop_id平均访问间隔
def get_shop_id_feat_2(start_date, end_date):
    dump_path = './cache/shop_id_feat2_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['shop_id', 'action_time']]
        df['action_time'] = df['action_time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['shop_id', 'action_time'], keep='first')
        df['action_time'] = df['action_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby('shop_id', as_index=False).agg(lambda x: x['action_time'].diff().mean())
        actions['avg_visit'] = actions['action_time'].dt.days
        del actions['action_time']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat2_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat2 finished')
    return actions


# shop_id四种行为平均访问间隔
def get_shop_id_feat_3(start_date, end_date):
    dump_path = './cache/shop_id_feat3_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['shop_id', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df['action_time'] = (pd.to_datetime(start_date) - pd.to_datetime(df['action_time'])).dt.days * (-1)
        df = df.drop_duplicates(['shop_id', 'action_time', 'type'], keep='first')
        actions = df.groupby(['shop_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat3_six_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat3 finished')
    return actions


# 最近K天
def shop_id_top_k_0_1(start_date, end_date):
    actions = get_actions_product(start_date, end_date)
    actions = actions[['user_id', 'shop_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby('shop_id', as_index=False).sum()
    del actions['type']
    del actions['user_id']
    shop_id = actions['shop_id']
    del actions['shop_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([shop_id, actions], axis=1)
    return actions


# 最近K天行为0/1提取
def get_shop_id_feat_4(start_date, end_date):
    dump_path = './cache/shop_id_feat4_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = shop_id_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, shop_id_top_k_0_1(start_days, end_date), how='outer', on='shop_id')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat4_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat4 finished')
    return actions


# 商品的重复购买率
def get_shop_id_feat_5(start_date, end_date):
    dump_path = './cache/shop_id_feat5_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'shop_id', 'type']]
        df = df[df['type'] == 2]  # 购买的行为
        df = df.groupby(['user_id', 'shop_id'], as_index=False).count()
        df.columns = ['user_id', 'shop_id', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['shop_id'], as_index=False)
        actions = grouped.count()[['shop_id', 'count1']]
        actions.columns = ['shop_id', 'count']
        re_count = grouped.sum()[['shop_id', 'count1']]
        re_count.columns = ['shop_id', 're_count']
        actions = pd.merge(actions, re_count, on='shop_id', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['shop_id'], re_buy_rate], axis=1)
        actions.columns = ['shop_id', 're_buy_rate']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat5_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat5 finished')
    return actions


# 获取货物最近一次行为的时间距离当前时间的差距
def get_shop_id_feat_6(start_date, end_date):
    dump_path = './cache/shop_id_feat6_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['shop_id', 'action_time', 'type']]
        df = df.drop_duplicates(['shop_id', 'type'], keep='last')
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days + 1
        actions = df.groupby(['shop_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat6_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat6 finished')
    return actions


# 商品购买/加入购物车/关注前访问次数
def get_shop_id_feat_7(start_date, end_date):
    dump_path = './cache/shop_id_feat7_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 商品购买前访问次数
        def shop_id_feat_7_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['shop_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('shop_id', as_index=False).count()
            visit.columns = ['shop_id', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('shop_id', as_index=False).count()
            buy.columns = ['shop_id', 'buy']
            actions = pd.merge(visit, buy, on='shop_id', how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问次数
        def shop_id_feat_7_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['shop_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('shop_id', as_index=False).count()
            visit.columns = ['shop_id', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('shop_id', as_index=False).count()
            guanzhu.columns = ['shop_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='shop_id', how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注次数
        def shop_id_feat_7_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['shop_id', 'type']]
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby('shop_id', as_index=False).count()
            guanzhu.columns = ['shop_id', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby('shop_id', as_index=False).count()
            buy.columns = ['shop_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='shop_id', how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(shop_id_feat_7_1(start_date, end_date), shop_id_feat_7_2(start_date, end_date), on='shop_id',
                           how='outer')
        actions = pd.merge(actions, shop_id_feat_7_3(start_date, end_date), on='shop_id', how='outer')
        shop_id = actions['shop_id']
        del actions['shop_id']
        actions = actions.fillna(0)
        actions = pd.concat([shop_id, pd.DataFrame(actions)], axis=1)

        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat7_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat7 finished')
    return actions


# 商品行为的交叉
def get_shop_id_feat_8(start_date, end_date):
    dump_path = './cache/shop_id_feat8_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)[['shop_id', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['shop_id', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby('shop_id', as_index=False).count()
        del action2['type']
        action2.columns = ['shop_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='shop_id')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat8_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat8 finished')
    return actions


# 获取最后一次行为的次数
def get_shop_id_feat_9(start_date, end_date):
    dump_path = './cache/shop_id_feat9_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:

        df = get_actions_product(start_date, end_date)[['shop_id', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days * (-1)
        idx = df.groupby(['shop_id', 'type'])['action_time'].transform(min)
        idx1 = idx == df['action_time']
        actions = df[idx1].groupby(['shop_id', "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user = actions[['shop_id']]
        del actions['shop_id']
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat9_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat9 finished')
    return actions


# 用户浏览 关注到购买的时间间隔
def get_shop_id_feat_10(start_date, end_date):
    dump_path = './cache/shop_id_feat10_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        actions_liulan = actions[actions['type'] == 1][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_liulan['time_liulan'] = actions_liulan[
            'action_time']
        actions_liulan = actions_liulan[['user_id', 'cate', 'shop_id', 'time_liulan']]
        actions_liulan = actions_liulan.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_guanzhu = actions[actions['type'] == 3][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_guanzhu['time_guanzhu'] = actions_guanzhu[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_guanzhu = actions_guanzhu[['user_id', 'cate', 'shop_id', 'time_guanzhu']]
        actions_guanzhu = actions_guanzhu.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_goumai = actions[actions['type'] == 2][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_goumai['time_goumai'] = actions_goumai[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'cate', 'shop_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='last')

        actions1 = pd.merge(actions_liulan, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions1['time_jiange'] = actions1['time_goumai'] - actions1['time_liulan']
        actions1 = actions1.drop(['user_id', 'cate', 'time_goumai', 'time_liulan'], axis=1)
        actions1['time_jiange'] = actions1['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min1 = actions1.groupby('shop_id').min().reset_index()
        actions_min1.columns = ['shop_id', 'time_min1']
        actions_max1 = actions1.groupby('shop_id').max().reset_index()
        actions_max1.columns = ['shop_id', 'time_max1']
        del actions1

        actions2 = pd.merge(actions_guanzhu, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions2['time_jiange'] = actions2['time_goumai'] - actions2['time_guanzhu']
        actions2 = actions2.drop(['user_id', 'cate', 'time_goumai', 'time_guanzhu'], axis=1)
        actions2['time_jiange'] = actions2['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min2 = actions2.groupby('shop_id').min().reset_index()
        actions_min2.columns = ['shop_id', 'time_min2']
        actions_max2 = actions2.groupby('shop_id').max().reset_index()
        actions_max2.columns = ['shop_id', 'time_max2']
        del actions2

        actions = pd.merge(actions_min1, actions_max1, on='shop_id', how='left')
        actions = pd.merge(actions, actions_min2, on='shop_id', how='left')
        actions = pd.merge(actions, actions_max2, on='shop_id', how='left')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['shop_id_feat10_' + str(i) for i in range(1, actions.shape[1])]
    print('shop_id feat10 finished')
    return actions


"""
#######################
F11类特征
#######################
"""


# F11购买/加入购物车/关注前访问天数
def get_F11_feat_1(start_date, end_date):
    dump_path = './cache/F11_feat1_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # shop_id购买前访问天数
        def shop_id_feat_1_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'cate', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby(['user_id', 'cate'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'buy']
            actions = pd.merge(visit, buy, on=['user_id', 'cate'], how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问天数
        def shop_id_feat_1_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'cate', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby(['user_id', 'cate'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby(['user_id', 'cate'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on=['user_id', 'cate'], how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注天数
        def shop_id_feat_1_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.drop_duplicates(['user_id', 'cate', 'action_time'], keep='first')
            del guanzhu['action_time']
            del actions['action_time']
            guanzhu = guanzhu.groupby(['user_id', 'cate'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'buy']
            actions = pd.merge(guanzhu, buy, on=['user_id', 'cate'], how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(shop_id_feat_1_1(start_date, end_date), shop_id_feat_1_2(start_date, end_date),
                           on=['user_id', 'cate'], how='outer')
        actions = pd.merge(actions, shop_id_feat_1_3(start_date, end_date), on=['user_id', 'cate'], how='outer')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat1_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat1 finished')
    return actions


# F11平均访问间隔
def get_F11_feat_2(start_date, end_date):
    dump_path = './cache/F11_feat2_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'action_time']]
        # df['action_time'] = df['action_time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['user_id', 'cate', 'action_time'], keep='first')
        df['action_time'] = df['action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby(['user_id', 'cate'], as_index=False).agg(lambda x: x['action_time'].diff().mean())
        actions['avg_visit'] = actions['action_time'].dt.days
        del actions['action_time']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['cate'] + ['F11_feat2_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# F11四种行为平均访问间隔
def get_F11_feat_3(start_date, end_date):
    dump_path = './cache/F11_feat3_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df['action_time'] = (pd.to_datetime(start_date) - pd.to_datetime(df['action_time'])).dt.days * (-1)
        df = df.drop_duplicates(['user_id', 'cate', 'action_time', 'type'], keep='first')
        actions = df.groupby(['user_id', 'cate', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat3_six_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat3 finished')
    return actions


# 最近K天
def F11_top_k_0_1(start_date, end_date):
    actions = get_actions_product(start_date, end_date)
    actions = actions[['user_id', 'cate', 'shop_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)
    actions = actions.groupby(['user_id', 'cate'], as_index=False).sum()
    del actions['type']
    shop_id = actions[['user_id', 'cate']]
    del actions['user_id']
    del actions['cate']
    del actions['shop_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([shop_id, actions], axis=1)
    return actions


# 最近K天行为0/1提取
def get_F11_feat_4(start_date, end_date):
    dump_path = './cache/F11_feat4_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = F11_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, F11_top_k_0_1(start_days, end_date), how='outer', on=['user_id', 'cate'])
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat4_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat4 finished')
    return actions


# 商品的重复购买率
def get_F11_feat_5(start_date, end_date):
    dump_path = './cache/F11_feat5_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type']]
        df = df[df['type'] == 2]  # 购买的行为
        df = df.groupby(['user_id', 'cate'], as_index=False).count()
        df.columns = ['user_id', 'cate', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['user_id', 'cate'], as_index=False)
        actions = grouped.count()[['user_id', 'cate', 'count1']]
        actions.columns = ['user_id', 'cate', 'count']
        re_count = grouped.sum()[['user_id', 'cate', 'count1']]
        re_count.columns = ['user_id', 'cate', 're_count']
        actions = pd.merge(actions, re_count, on=['user_id', 'cate'], how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions[['user_id', 'cate']], re_buy_rate], axis=1)
        actions.columns = ['user_id', 'cate', 're_buy_rate']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat5_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat5 finished')
    return actions


# 获取货物最近一次行为的时间距离当前时间的差距
def get_F11_feat_6(start_date, end_date):
    dump_path = './cache/F11_feat6_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'action_time', 'type']]
        df = df.drop_duplicates(['user_id', 'cate', 'type'], keep='last')
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days + 1
        actions = df.groupby(['user_id', 'cate', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat6_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat6 finished')
    return actions


# 商品购买/加入购物车/关注前访问次数
def get_F11_feat_7(start_date, end_date):
    dump_path = './cache/F11_feat7_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 商品购买前访问次数
        def shop_id_feat_7_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby(['user_id', 'cate'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'buy']
            actions = pd.merge(visit, buy, on=['user_id', 'cate'], how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问次数
        def shop_id_feat_7_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby(['user_id', 'cate'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby(['user_id', 'cate'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on=['user_id', 'cate'], how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注次数
        def shop_id_feat_7_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type']]
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby(['user_id', 'cate'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'buy']
            actions = pd.merge(guanzhu, buy, on=['user_id', 'cate'], how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(shop_id_feat_7_1(start_date, end_date), shop_id_feat_7_2(start_date, end_date),
                           on=['user_id', 'cate'],
                           how='outer')
        actions = pd.merge(actions, shop_id_feat_7_3(start_date, end_date), on=['user_id', 'cate'], how='outer')
        shop_id = actions[['user_id', 'cate']]
        del actions['user_id']
        del actions['cate']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([shop_id, pd.DataFrame(actions)], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat7_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat7 finished')
    return actions


# 商品行为的交叉
def get_F11_feat_8(start_date, end_date):
    dump_path = './cache/F11_feat8_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['user_id', 'cate', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby(['user_id', 'cate'], as_index=False).count()
        del action2['type']
        action2.columns = ['user_id', 'cate', 'cnt']
        actions = pd.merge(action1, action2, how='left', on=['user_id', 'cate'])
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat8_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat8 finished')
    return actions


# 获取最后一次行为的次数
def get_F11_feat_9(start_date, end_date):
    dump_path = './cache/F11_feat9_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:

        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days + 1
        idx = df.groupby(['user_id', 'cate', 'type'])['action_time'].transform(min)
        idx1 = idx == df['action_time']
        actions = df[idx1].groupby(['user_id', 'cate', "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user = actions[['user_id', 'cate']]
        del actions['user_id']
        del actions['cate']
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat9_' + str(i) for i in range(1, actions.shape[1] - 1
                                                                                  )]
    print('F11 feat9 finished')
    return actions


# 用户浏览 关注到购买的时间间隔
def get_F11_feat_10(start_date, end_date):
    dump_path = './cache/F11_feat10_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        actions_liulan = actions[actions['type'] == 1][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_liulan['time_liulan'] = actions_liulan[
            'action_time']
        actions_liulan = actions_liulan[['user_id', 'cate', 'shop_id', 'time_liulan']]
        actions_liulan = actions_liulan.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_guanzhu = actions[actions['type'] == 3][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_guanzhu['time_guanzhu'] = actions_guanzhu[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_guanzhu = actions_guanzhu[['user_id', 'cate', 'shop_id', 'time_guanzhu']]
        actions_guanzhu = actions_guanzhu.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_goumai = actions[actions['type'] == 2][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_goumai['time_goumai'] = actions_goumai[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'cate', 'shop_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='last')

        actions1 = pd.merge(actions_liulan, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions1['time_jiange'] = actions1['time_goumai'] - actions1['time_liulan']
        actions1 = actions1.drop(['shop_id', 'time_goumai', 'time_liulan'], axis=1)
        actions1['time_jiange'] = actions1['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min1 = actions1.groupby(['user_id', 'cate']).min().reset_index()
        actions_min1.columns = ['user_id', 'cate', 'time_min1']
        actions_max1 = actions1.groupby(['user_id', 'cate']).max().reset_index()
        actions_max1.columns = ['user_id', 'cate', 'time_max1']
        del actions1

        actions2 = pd.merge(actions_guanzhu, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions2['time_jiange'] = actions2['time_goumai'] - actions2['time_guanzhu']
        actions2 = actions2.drop(['shop_id', 'time_goumai', 'time_guanzhu'], axis=1)
        actions2['time_jiange'] = actions2['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min2 = actions2.groupby(['user_id', 'cate']).min().reset_index()
        actions_min2.columns = ['user_id', 'cate', 'time_min2']
        actions_max2 = actions2.groupby(['user_id', 'cate']).max().reset_index()
        actions_max2.columns = ['user_id', 'cate', 'time_max2']
        del actions2

        actions = pd.merge(actions_min1, actions_max1, on=['user_id', 'cate'], how='left')
        actions = pd.merge(actions, actions_min2, on=['user_id', 'cate'], how='left')
        actions = pd.merge(actions, actions_max2, on=['user_id', 'cate'], how='left')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat10_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat10 finished')
    return actions


# 用户总购买/关注/浏览/评论品牌数
def get_F11_feat_11(start_date, end_date):
    dump_path = './cache/F11_feat11_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        action = None
        for i in (1, 2, 3, 4):
            df = actions[actions['type'] == i][['user_id', 'cate', 'shop_id']]
            df = df.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')
            df = df.groupby(['user_id', 'cate'], as_index=False).count()
            df.columns = ['user_id', 'cate', 'num_%s' % i]
            if i == 1:
                action = df
            else:
                action = pd.merge(action, df, on=['user_id', 'cate'], how='outer')
        actions = action.fillna(0)
        actions = actions.astype('float')
        user = actions[['user_id', 'cate']]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.drop(['user_id', 'cate'], axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat11_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat11 finished')
    return actions


# 层级的天数
def get_F11_feat_12(start_date, end_date, n):
    dump_path = './cache/F11_feat12_F12_7_%s_%s_%s.pkl' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days // n
        df = df.drop_duplicates(['user_id', 'cate', 'type', 'action_time'], keep='first')
        actions = df.groupby(['user_id', 'cate', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['user_id', 'cate']]
        del actions['user_id']
        del actions['cate']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat12_' + str(n) + '_' + str(i) for i in
                                             range(1, actions.shape[1] - 1)]
    print('F11 feat12 finished')
    return actions


# 用户每隔7天购买次数
def get_F11_feat_13(start_date, end_date):
    dump_path = './cache/F11_feat13_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        n = 7
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'action_time', 'type']]
        df = df[df['type'] == 2][['user_id', 'cate', 'action_time']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days // n
        df['cnt'] = 0
        actions = df.groupby(['user_id', 'cate', 'action_time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['user_id', 'cate']]
        del actions['user_id']
        del actions['cate']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['F11_feat13_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('F11 feat13 finished')
    return actions


"""
#######################
F12类特征
#######################
"""
F12 = ['user_id', 'cate', 'shop_id']


# F12购买/加入购物车/关注前访问天数
def get_F12_feat_1(start_date, end_date):
    dump_path = './cache/F12_feat1_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # shop_id购买前访问天数
        def F12_feat_1_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'cate', 'shop_id', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'shop_id', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'shop_id', 'buy']
            actions = pd.merge(visit, buy, on=['user_id', 'cate', 'shop_id'], how='left')
            actions['F12_visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问天数
        def F12_feat_1_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'cate', 'shop_id', 'action_time'], keep='first')
            del visit['action_time']
            del actions['action_time']
            visit = visit.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'shop_id', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'shop_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on=['user_id', 'cate', 'shop_id'], how='left')
            actions['F12_visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注天数
        def F12_feat_1_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type', 'action_time']]
            actions['action_time'] = actions['action_time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.drop_duplicates(['user_id', 'cate', 'shop_id', 'action_time'], keep='first')
            del guanzhu['action_time']
            del actions['action_time']
            guanzhu = guanzhu.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'shop_id', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'shop_id', 'buy']
            actions = pd.merge(guanzhu, buy, on=['user_id', 'cate', 'shop_id'], how='left')
            actions['F12_guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(F12_feat_1_1(start_date, end_date), F12_feat_1_2(start_date, end_date),
                           on=['user_id', 'cate', 'shop_id'], how='outer')
        actions = pd.merge(actions, F12_feat_1_3(start_date, end_date), on=['user_id', 'cate', 'shop_id'], how='outer')
        pickle.dump(actions, open(dump_path, 'wb'))
    print('F12_feat1 finished')
    return actions


# F12_平均访问间隔
def get_F12_feat_2(start_date, end_date):
    dump_path = './cache/F12_feat2_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'action_time']]
        # df['action_time'] = df['action_time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['user_id', 'cate', 'shop_id', 'action_time'], keep='first')
        actions = df.groupby(['user_id', 'cate', 'shop_id'], as_index=False).agg(
            lambda x: x['action_time'].diff().mean())
        actions['F12_avg_visit'] = actions['action_time'].dt.days
        del actions['action_time']
        pickle.dump(actions, open(dump_path, 'wb'))
    print('F12 feat2 finished')
    return actions


# F11四种行为平均访问间隔
def get_F12_feat_3(start_date, end_date):
    dump_path = './cache/F12_feat3_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df['action_time'] = (pd.to_datetime(start_date) - pd.to_datetime(df['action_time'])).dt.days * (-1)
        df = df.drop_duplicates(['user_id', 'cate', 'shop_id', 'action_time', 'type'], keep='first')
        actions = df.groupby(['user_id', 'cate', 'shop_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        user = actions[['user_id', 'cate', 'shop_id']]
        actions = actions.drop(['user_id', 'cate', 'shop_id'], axis=1)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate', 'shop_id'] + ['F12_feat3_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat3 finished')
    return actions


# 最近K天
def F12_top_k_0_1(start_date, end_date):
    actions = get_actions_product(start_date, end_date)
    actions = actions[['user_id', 'cate', 'shop_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)
    actions = actions.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
    del actions['type']
    shop_id = actions[['user_id', 'cate', 'shop_id']]
    del actions['user_id']
    del actions['cate']
    del actions['shop_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([shop_id, actions], axis=1)
    return actions


# 最近K天行为0/1提取
def get_F12_feat_4(start_date, end_date):
    dump_path = './cache/F12_feat4_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = F12_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, F12_top_k_0_1(start_days, end_date), how='outer',
                                   on=['user_id', 'cate', 'shop_id'])
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = F12 + ['F12_feat4_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat4 finished')
    return actions


# 商品的重复购买率
def get_F12_feat_5(start_date, end_date):
    dump_path = './cache/F12_feat5_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type']]
        df = df[df['type'] == 2]  # 购买的行为
        df = df.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
        df.columns = ['user_id', 'cate', 'shop_id', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['user_id', 'cate', 'shop_id'], as_index=False)
        actions = grouped.count()[['user_id', 'cate', 'shop_id', 'count1']]
        actions.columns = ['user_id', 'cate', 'shop_id', 'count']
        re_count = grouped.sum()[['user_id', 'cate', 'shop_id', 'count1']]
        re_count.columns = ['user_id', 'cate', 'shop_id', 're_count']
        actions = pd.merge(actions, re_count, on=['user_id', 'cate', 'shop_id'], how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions[['user_id', 'cate', 'shop_id']], re_buy_rate], axis=1)
        actions.columns = ['user_id', 'cate', 'shop_id', 're_buy_rate']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = F12 + ['F12_feat5_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat5 finished')
    return actions


# 获取货物最近一次行为的时间距离当前时间的差距
def get_F12_feat_6(start_date, end_date):
    dump_path = './cache/F12_feat6_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'action_time', 'type']]
        df = df.drop_duplicates(['user_id', 'cate', 'shop_id', 'type'], keep='last')
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days + 1
        actions = df.groupby(['user_id', 'cate', 'shop_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = F12 + ['F12_feat6_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat6 finished')
    return actions


# 商品购买/加入购物车/关注前访问次数
def get_F12_feat_7(start_date, end_date):
    dump_path = './cache/F12_feat7_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 商品购买前访问次数
        def shop_id_feat_7_1(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'shop_id', 'visit']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'shop_id', 'buy']
            actions = pd.merge(visit, buy, on=['user_id', 'cate', 'shop_id'], how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品关注前访问次数
        def shop_id_feat_7_2(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            visit.columns = ['user_id', 'cate', 'shop_id', 'visit']
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'shop_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on=['user_id', 'cate', 'shop_id'], how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前关注次数
        def shop_id_feat_7_3(start_date, end_date):
            actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type']]
            guanzhu = actions[actions['type'] == 3]
            guanzhu = guanzhu.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            guanzhu.columns = ['user_id', 'cate', 'shop_id', 'guanzhu']
            buy = actions[actions['type'] == 2]
            buy = buy.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
            buy.columns = ['user_id', 'cate', 'shop_id', 'buy']
            actions = pd.merge(guanzhu, buy, on=['user_id', 'cate', 'shop_id'], how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(shop_id_feat_7_1(start_date, end_date), shop_id_feat_7_2(start_date, end_date),
                           on=['user_id', 'cate', 'shop_id'],
                           how='outer')
        actions = pd.merge(actions, shop_id_feat_7_3(start_date, end_date), on=['user_id', 'cate', 'shop_id'],
                           how='outer')
        shop_id = actions[['user_id', 'cate', 'shop_id']]
        del actions['user_id']
        del actions['cate']
        del actions['shop_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([shop_id, pd.DataFrame(actions)], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = F12 + ['F12_feat7_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat7 finished')
    return actions


# 商品行为的交叉
def get_F12_feat_8(start_date, end_date):
    dump_path = './cache/F12_feat8_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['user_id', 'cate', 'shop_id', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby(['user_id', 'cate', 'shop_id'], as_index=False).count()
        del action2['type']
        action2.columns = ['user_id', 'cate', 'shop_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on=['user_id', 'cate', 'shop_id'])
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = F12 + ['F12_feat8_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat8 finished')
    return actions


# 获取最后一次行为的次数
def get_F12_feat_9(start_date, end_date):
    dump_path = './cache/F12_feat9_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:

        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'shop_id', 'action_time', 'type']]
        # df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        df['action_time'] = (pd.to_datetime(end_date) - pd.to_datetime(df['action_time'])).dt.days + 1
        idx = df.groupby(['user_id', 'cate', 'shop_id', 'type'])['action_time'].transform(min)
        idx1 = idx == df['action_time']
        actions = df[idx1].groupby(['user_id', 'cate', 'shop_id', "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user = actions[['user_id', 'cate', 'shop_id']]
        del actions['user_id']
        del actions['cate']
        del actions['shop_id']
        actions = pd.DataFrame(actions)
        actions = pd.concat([user, actions], axis=1)
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate', 'shop_id'] + ['F12_feat9_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat9 finished')
    return actions


# 用户浏览 关注到购买的时间间隔
def get_F12_feat_10(start_date, end_date):
    dump_path = './cache/F12_feat10_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        actions_liulan = actions[actions['type'] == 1][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_liulan['time_liulan'] = actions_liulan[
            'action_time']
        actions_liulan = actions_liulan[['user_id', 'cate', 'shop_id', 'time_liulan']]
        actions_liulan = actions_liulan.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_guanzhu = actions[actions['type'] == 3][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_guanzhu['time_guanzhu'] = actions_guanzhu[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_guanzhu = actions_guanzhu[['user_id', 'cate', 'shop_id', 'time_guanzhu']]
        actions_guanzhu = actions_guanzhu.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='first')

        actions_goumai = actions[actions['type'] == 2][['user_id', 'cate', 'shop_id', 'action_time']]
        actions_goumai['time_goumai'] = actions_goumai[
            'action_time']  # .map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'cate', 'shop_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='last')

        actions1 = pd.merge(actions_liulan, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions1['time_jiange'] = actions1['time_goumai'] - actions1['time_liulan']
        actions1 = actions1.drop(['time_goumai', 'time_liulan'], axis=1)
        actions1['time_jiange'] = actions1['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min1 = actions1.groupby(['user_id', 'cate', 'shop_id']).min().reset_index()
        actions_min1.columns = ['user_id', 'cate', 'shop_id', 'time_min1']
        actions_max1 = actions1.groupby(['user_id', 'cate', 'shop_id']).max().reset_index()
        actions_max1.columns = ['user_id', 'cate', 'shop_id', 'time_max1']
        del actions1

        actions2 = pd.merge(actions_guanzhu, actions_goumai, on=['user_id', 'cate', 'shop_id'], how='inner')
        actions2['time_jiange'] = actions2['time_goumai'] - actions2['time_guanzhu']
        actions2 = actions2.drop(['time_goumai', 'time_guanzhu'], axis=1)
        actions2['time_jiange'] = actions2['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min2 = actions2.groupby(['user_id', 'cate', 'shop_id']).min().reset_index()
        actions_min2.columns = ['user_id', 'cate', 'shop_id', 'time_min2']
        actions_max2 = actions2.groupby(['user_id', 'cate', 'shop_id']).max().reset_index()
        actions_max2.columns = ['user_id', 'cate', 'shop_id', 'time_max2']
        del actions2

        actions = pd.merge(actions_min1, actions_max1, on=['user_id', 'cate', 'shop_id'], how='left')
        actions = pd.merge(actions, actions_min2, on=['user_id', 'cate', 'shop_id'], how='left')
        actions = pd.merge(actions, actions_max2, on=['user_id', 'cate', 'shop_id'], how='left')
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate', 'shop_id'] + ['F12_feat10_' + str(i) for i in range(1, actions.shape[1] - 2)]
    print('F12 feat10 finished')
    return actions


# 标签
def get_labels(start_date, end_date):
    dump_path = './cache/labels_F12_7_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions = actions[actions['type'] == 2]
        actions = actions.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'cate', 'shop_id', 'label']]
        actions.to_pickle(dump_path)
    print('label finished')
    return actions


def make_train_set_F12_7(train_start_date, train_end_date, test_start_date, test_end_date, start):
    dump_path = './cache/train_set_F12_7_%s_%s_%s_%s.pkl' % (
        train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 索引
        f11_actions = get_actions_product(train_start_date, train_end_date)
        f11_actions = f11_actions.drop_duplicates(['user_id', 'cate', 'shop_id'])
        f11_actions = f11_actions[['user_id', 'cate', 'shop_id']]

        # 标签
        labels = get_labels(test_start_date, test_end_date)

        # 特征
        start_days = "2018-02-01"                              #
        user = get_basic_user_feat()
        shop = get_shop_feat(start_days, train_end_date)
        product_stat = get_product_stat_feat(start_days, train_end_date)
        time = get_time_feat(start_days, train_end_date)
        stat_feat = get_stat_feat(start_days, train_end_date)
        user_feat = user_features(start_days, train_end_date)
        cross_feat = get_cross_feat(start_days, train_end_date)

        # user
        user_feat1 = get_user_feat1(start_days, train_end_date)
        user_feat2 = get_user_feat2(start_days, train_end_date)
        user_feat3 = get_user_feat3(start_days, train_end_date)
        user_feat5 = get_user_feat5(start_days, train_end_date)
        user_feat6 = get_user_feat6(start_days, train_end_date)
        user_feat7 = get_user_feat7(start_days, train_end_date)
        user_feat8 = get_user_feat8(start_days, train_end_date)
        user_feat9 = get_user_feat9(start_days, train_end_date)
        user_feat10 = get_user_feat10(start_days, train_end_date)
        user_feat11 = get_user_feat11(start_days, train_end_date)
        user_feat12 = get_user_feat12(start_days, train_end_date)
        user_feat13 = get_user_feat13(start_days, train_end_date)
        user_feat14 = get_user_feat14(start_days, train_end_date)
        user_feat15 = get_user_feat15(start_days, train_end_date)           #

        cate_feat1 = get_cate_feat_1(start_days, train_end_date)
        cate_feat2 = get_cate_feat_2(start_days, train_end_date)
        cate_feat3 = get_cate_feat_3(start_days, train_end_date)
        cate_feat4 = get_cate_feat_4(start_days, train_end_date)
        cate_feat5 = get_cate_feat_5(start_days, train_end_date)
        cate_feat6 = get_cate_feat_6(start_days, train_end_date)
        cate_feat7 = get_cate_feat_7(start_days, train_end_date)
        cate_feat8 = get_cate_feat_8(start_days, train_end_date)
        cate_feat9 = get_cate_feat_9(start_days, train_end_date)
        cate_feat10 = get_cate_feat_10(start_days, train_end_date)
        cate_feat11 = get_cate_feat_11(start_days, train_end_date)

        shop_id_feat1 = get_shop_id_feat_1(start_days, train_end_date)
        shop_id_feat2 = get_shop_id_feat_2(start_days, train_end_date)
        shop_id_feat3 = get_shop_id_feat_3(start_days, train_end_date)
        shop_id_feat4 = get_shop_id_feat_4(start_days, train_end_date)
        shop_id_feat5 = get_shop_id_feat_5(start_days, train_end_date)
        shop_id_feat6 = get_shop_id_feat_6(start_days, train_end_date)
        shop_id_feat7 = get_shop_id_feat_7(start_days, train_end_date)
        shop_id_feat8 = get_shop_id_feat_8(start_days, train_end_date)

        F11_feat1 = get_F11_feat_1(start_days, train_end_date)
        F11_feat3 = get_F11_feat_3(start_days, train_end_date)
        F11_feat4 = get_F11_feat_4(start_days, train_end_date)
        F11_feat5 = get_F11_feat_5(start_days, train_end_date)
        F11_feat6 = get_F11_feat_6(start_days, train_end_date)
        F11_feat7 = get_F11_feat_7(start_days, train_end_date)
        F11_feat8 = get_F11_feat_8(start_days, train_end_date)
        F11_feat9 = get_F11_feat_9(start_days, train_end_date)
        F11_feat10 = get_F11_feat_10(start_days, train_end_date)
        F11_feat11 = get_F11_feat_11(start_days, train_end_date)

        F12_feat1 = get_F12_feat_1(start_days, train_end_date)
        F12_feat3 = get_F12_feat_3(start_days, train_end_date)
        F12_feat4 = get_F12_feat_4(start_days, train_end_date)
        F12_feat5 = get_F12_feat_5(start_days, train_end_date)
        F12_feat6 = get_F12_feat_6(start_days, train_end_date)
        F12_feat7 = get_F12_feat_7(start_days, train_end_date)
        F12_feat8 = get_F12_feat_8(start_days, train_end_date)
        F12_feat9 = get_F12_feat_9(start_days, train_end_date)
        F12_feat10 = get_F12_feat_10(start_days, train_end_date)

        # 滑窗行为特征
        actions = None
        for i in (3, 5, 7, 14, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_accumulate_user_feat(start_days, train_end_date)
            else:
                actions1 = get_accumulate_user_feat(start_days, train_end_date)
                actions = pd.merge(actions, actions1, how='left', on=['user_id', 'cate', 'shop_id'])

        # 前一天滑窗行为 包含cart
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=1)
        start_days = start_days.strftime('%Y-%m-%d')
        actions_cart = get_accumulate_user_feat_cart(start_days, train_end_date)

        # act_5
        # act5_feat = pd.read_csv('./cache/train_lastday_act5_stat.csv')
        act5_feat = get_last1day_cart_fearture(start_days, train_end_date, 1)

        # 负采样
        f11_actions = pd.merge(f11_actions, labels, how='left', on=['user_id', 'cate', 'shop_id'])
        f11_actions = f11_actions.fillna(0)
        print('train data size:', f11_actions.shape[0])
        f11_actions_1 = f11_actions[f11_actions['label'] == 1]
        f11_actions_0 = f11_actions[f11_actions['label'] == 0]
        frac1 = (f11_actions_1.shape[0] * 30) / f11_actions_0.shape[0]  # 负样本为正样本30倍
        f11_actions_0 = f11_actions_0.sample(frac=frac1).reset_index(drop=True)
        f11_actions = pd.concat([f11_actions_1, f11_actions_0], axis=0, ignore_index=True)
        f11_actions = f11_actions.sample(frac=1).reset_index(drop=True)
        print('train data size after sample:', f11_actions.shape[0])

        actions = pd.merge(f11_actions, actions, how='left', on=['user_id', 'cate', 'shop_id'])
        # actions = pd.merge(actions, actions_cart, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, time, how='left', on='user_id')
        actions = pd.merge(actions, stat_feat, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, shop, how='left', on='shop_id')
        actions = pd.merge(actions, product_stat, how='left', on='cate')

        actions = pd.merge(actions, user_feat1, how='left', on='user_id')
        actions = pd.merge(actions, user_feat2, how='left', on='user_id')
        actions = pd.merge(actions, user_feat3, how='left', on='user_id')
        actions = pd.merge(actions, user_feat5, how='left', on='user_id')
        actions = pd.merge(actions, user_feat6, how='left', on='user_id')
        actions = pd.merge(actions, user_feat7, how='left', on='user_id')
        actions = pd.merge(actions, user_feat8, how='left', on='user_id')
        actions = pd.merge(actions, user_feat9, how='left', on='user_id')
        actions = pd.merge(actions, user_feat10, how='left', on='user_id')
        actions = pd.merge(actions, user_feat11, how='left', on='user_id')
        actions = pd.merge(actions, user_feat12, how='left', on='user_id')
        actions = pd.merge(actions, user_feat13, how='left', on='user_id')
        actions = pd.merge(actions, user_feat14, how='left', on='user_id')
        actions = pd.merge(actions, user_feat, how='left', on='user_id')
        actions = pd.merge(actions, user_feat15, how='left', on=['user_id', 'cate', 'shop_id'])

        """
        cate
        """
        actions = pd.merge(actions, cate_feat1, how='left', on='cate')
        actions = pd.merge(actions, cate_feat2, how='left', on='cate')
        actions = pd.merge(actions, cate_feat3, how='left', on='cate')
        actions = pd.merge(actions, cate_feat4, how='left', on='cate')
        actions = pd.merge(actions, cate_feat5, how='left', on='cate')
        actions = pd.merge(actions, cate_feat6, how='left', on='cate')
        actions = pd.merge(actions, cate_feat7, how='left', on='cate')
        actions = pd.merge(actions, cate_feat8, how='left', on='cate')
        actions = pd.merge(actions, cate_feat9, how='left', on='cate')
        actions = pd.merge(actions, cate_feat10, how='left', on='cate')
        actions = pd.merge(actions, cate_feat11, how='left', on='cate')
        print('cate finished')
        """
        shop_id
        """
        actions = pd.merge(actions, shop_id_feat1, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat2, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat3, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat4, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat5, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat6, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat7, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat8, how='left', on='shop_id')
        print('shop finished')
        """
        F11
        """
        actions = pd.merge(actions, F11_feat1, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat3, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat4, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat5, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat6, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat7, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat8, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat9, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat10, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat11, how='left', on=['user_id', 'cate'])
        print('F11 finished')
        """
        F12
        """
        actions = pd.merge(actions, F12_feat1, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat3, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat4, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat5, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat6, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat7, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat8, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat9, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat10, how='left', on=['user_id', 'cate', 'shop_id'])
        print('F12 finished')

        actions = pd.merge(actions, cross_feat, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, act5_feat, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, actions_cart, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = actions.fillna(0)
        # actions.to_pickle(dump_path)
    print('train_set finised')
    return actions


def make_test_set_F12_7(train_start_date, train_end_date,start):
    dump_path = './cache/test_set_F12_7_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 索引
        f11_actions = get_actions_product(train_start_date, train_end_date)
        f11_actions = f11_actions.drop_duplicates(['user_id', 'cate', 'shop_id'])
        f11_actions = f11_actions[['user_id', 'cate', 'shop_id']]  #

        # 特征
        start_days = "2018-02-01"           # "2018-02-01"
        user = get_basic_user_feat()
        shop = get_shop_feat(start_days, train_end_date)
        product_stat = get_product_stat_feat(start_days, train_end_date)
        time = get_time_feat(start_days, train_end_date)
        stat_feat = get_stat_feat(start_days, train_end_date)
        user_feat = user_features(start_days, train_end_date)
        cross_feat = get_cross_feat(start_days, train_end_date)

        user_feat1 = get_user_feat1(start_days, train_end_date)
        user_feat2 = get_user_feat2(start_days, train_end_date)
        user_feat3 = get_user_feat3(start_days, train_end_date)
        user_feat5 = get_user_feat5(start_days, train_end_date)
        user_feat6 = get_user_feat6(start_days, train_end_date)
        user_feat7 = get_user_feat7(start_days, train_end_date)
        user_feat8 = get_user_feat8(start_days, train_end_date)
        user_feat9 = get_user_feat9(start_days, train_end_date)
        user_feat10 = get_user_feat10(start_days, train_end_date)
        user_feat11 = get_user_feat11(start_days, train_end_date)
        user_feat12 = get_user_feat12(start_days, train_end_date)
        user_feat13 = get_user_feat13(start_days, train_end_date)
        user_feat14 = get_user_feat14(start_days, train_end_date)
        user_feat15 = get_user_feat15(start_days, train_end_date)      #

        cate_feat1 = get_cate_feat_1(start_days, train_end_date)
        cate_feat2 = get_cate_feat_2(start_days, train_end_date)
        cate_feat3 = get_cate_feat_3(start_days, train_end_date)
        cate_feat4 = get_cate_feat_4(start_days, train_end_date)
        cate_feat5 = get_cate_feat_5(start_days, train_end_date)
        cate_feat6 = get_cate_feat_6(start_days, train_end_date)
        cate_feat7 = get_cate_feat_7(start_days, train_end_date)
        cate_feat8 = get_cate_feat_8(start_days, train_end_date)
        cate_feat9 = get_cate_feat_9(start_days, train_end_date)
        cate_feat10 = get_cate_feat_10(start_days, train_end_date)
        cate_feat11 = get_cate_feat_11(start_days, train_end_date)

        shop_id_feat1 = get_shop_id_feat_1(start_days, train_end_date)
        shop_id_feat2 = get_shop_id_feat_2(start_days, train_end_date)
        shop_id_feat3 = get_shop_id_feat_3(start_days, train_end_date)
        shop_id_feat4 = get_shop_id_feat_4(start_days, train_end_date)
        shop_id_feat5 = get_shop_id_feat_5(start_days, train_end_date)
        shop_id_feat6 = get_shop_id_feat_6(start_days, train_end_date)
        shop_id_feat7 = get_shop_id_feat_7(start_days, train_end_date)
        shop_id_feat8 = get_shop_id_feat_8(start_days, train_end_date)

        F11_feat1 = get_F11_feat_1(start_days, train_end_date)
        F11_feat3 = get_F11_feat_3(start_days, train_end_date)
        F11_feat4 = get_F11_feat_4(start_days, train_end_date)
        F11_feat5 = get_F11_feat_5(start_days, train_end_date)
        F11_feat6 = get_F11_feat_6(start_days, train_end_date)
        F11_feat7 = get_F11_feat_7(start_days, train_end_date)
        F11_feat8 = get_F11_feat_8(start_days, train_end_date)
        F11_feat9 = get_F11_feat_9(start_days, train_end_date)
        F11_feat10 = get_F11_feat_10(start_days, train_end_date)
        F11_feat11 = get_F11_feat_11(start_days, train_end_date)

        F12_feat1 = get_F12_feat_1(start_days, train_end_date)
        F12_feat3 = get_F12_feat_3(start_days, train_end_date)
        F12_feat4 = get_F12_feat_4(start_days, train_end_date)
        F12_feat5 = get_F12_feat_5(start_days, train_end_date)
        F12_feat6 = get_F12_feat_6(start_days, train_end_date)
        F12_feat7 = get_F12_feat_7(start_days, train_end_date)
        F12_feat8 = get_F12_feat_8(start_days, train_end_date)
        F12_feat9 = get_F12_feat_9(start_days, train_end_date)
        F12_feat10 = get_F12_feat_10(start_days, train_end_date)

        # generate 时间窗口
        actions = None
        for i in (3, 5, 7, 14, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_accumulate_user_feat(start_days, train_end_date)
            else:
                actions1 = get_accumulate_user_feat(start_days, train_end_date)
                actions = pd.merge(actions, actions1, how='left', on=['user_id', 'cate', 'shop_id'])

        # 前一天滑窗行为 包含cart
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=1)
        start_days = start_days.strftime('%Y-%m-%d')
        actions_cart = get_accumulate_user_feat_cart(start_days, train_end_date)

        # act_5
        # act5_feat = pd.read_csv('./cache/test_lastday_act5_stat.csv')
        act5_feat = get_last1day_cart_fearture(start_days, train_end_date, 1)

        actions = pd.merge(f11_actions, actions, how='left', on=['user_id', 'cate', 'shop_id'])
        # actions = pd.merge(actions, actions_cart, how='left', on=['user_id', 'cate', 'shop_id'])   #
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, time, how='left', on='user_id')
        actions = pd.merge(actions, stat_feat, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, shop, how='left', on='shop_id')
        actions = pd.merge(actions, product_stat, how='left', on='cate')

        actions = pd.merge(actions, user_feat1, how='left', on='user_id')
        actions = pd.merge(actions, user_feat2, how='left', on='user_id')
        actions = pd.merge(actions, user_feat3, how='left', on='user_id')
        actions = pd.merge(actions, user_feat5, how='left', on='user_id')
        actions = pd.merge(actions, user_feat6, how='left', on='user_id')
        actions = pd.merge(actions, user_feat7, how='left', on='user_id')
        actions = pd.merge(actions, user_feat8, how='left', on='user_id')
        actions = pd.merge(actions, user_feat9, how='left', on='user_id')
        actions = pd.merge(actions, user_feat10, how='left', on='user_id')
        actions = pd.merge(actions, user_feat11, how='left', on='user_id')
        actions = pd.merge(actions, user_feat12, how='left', on='user_id')
        actions = pd.merge(actions, user_feat13, how='left', on='user_id')
        actions = pd.merge(actions, user_feat14, how='left', on='user_id')
        actions = pd.merge(actions, user_feat, how='left', on='user_id')
        actions = pd.merge(actions, user_feat15, how='left', on=['user_id', 'cate', 'shop_id'])
        """
        cate
        """
        actions = pd.merge(actions, cate_feat1, how='left', on='cate')
        actions = pd.merge(actions, cate_feat2, how='left', on='cate')
        actions = pd.merge(actions, cate_feat3, how='left', on='cate')
        actions = pd.merge(actions, cate_feat4, how='left', on='cate')
        actions = pd.merge(actions, cate_feat5, how='left', on='cate')
        actions = pd.merge(actions, cate_feat6, how='left', on='cate')
        actions = pd.merge(actions, cate_feat7, how='left', on='cate')
        actions = pd.merge(actions, cate_feat8, how='left', on='cate')
        actions = pd.merge(actions, cate_feat9, how='left', on='cate')
        actions = pd.merge(actions, cate_feat10, how='left', on='cate')
        actions = pd.merge(actions, cate_feat11, how='left', on='cate')
        print('cate finished')
        """
        shop_id
        """
        actions = pd.merge(actions, shop_id_feat1, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat2, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat3, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat4, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat5, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat6, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat7, how='left', on='shop_id')
        actions = pd.merge(actions, shop_id_feat8, how='left', on='shop_id')
        print('shop finished')
        """
        F11
        """
        actions = pd.merge(actions, F11_feat1, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat3, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat4, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat5, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat6, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat7, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat8, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat9, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat10, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, F11_feat11, how='left', on=['user_id', 'cate'])
        print('F11 finished')
        """
        F12
        """
        actions = pd.merge(actions, F12_feat1, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat3, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat4, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat5, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat6, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat7, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat8, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat9, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, F12_feat10, how='left', on=['user_id', 'cate', 'shop_id'])
        print('F12 finished')

        actions = pd.merge(actions, cross_feat, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, act5_feat, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, actions_cart, how='left', on=['user_id', 'cate', 'shop_id'])

        actions = actions.fillna(0)
        # actions.to_pickle(dump_path)
        del stat_feat, f11_actions
    print('test_set finished')
    return actions


def lgb_train_F12_7(X_train1, y_train1, X_test1, sub_user_index):
    # 提交结果
    sub = sub_user_index[['user_id', 'cate', 'shop_id']].copy()
    sub['label'] = 0

    # 训练测试集
    X_train = X_train1.values
    y_train = y_train1.values
    X_test = X_test1.values

    del X_train1, y_train1, X_test1

    print('================================')
    print(X_train.shape)
    print(X_test.shape)
    print('================================')

    xx_logloss = []
    oof_preds = np.zeros(X_train.shape[0])
    N = 5
    skf = StratifiedKFold(n_splits=N, random_state=1024, shuffle=True)

    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1,
    }
    for k, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print('train _K_ flod', k)

        lgb_train = lgb.Dataset(X_train[train_index], y_train[train_index])
        lgb_evals = lgb.Dataset(X_train[test_index], y_train[test_index], reference=lgb_train)

        lgbm = lgb.train(params, lgb_train, num_boost_round=50000, valid_sets=[lgb_train, lgb_evals],
                         valid_names=['train', 'valid'], early_stopping_rounds=100, verbose_eval=200)

        sub['label'] += lgbm.predict(X_test, num_iteration=lgbm.best_iteration) / N
        oof_preds[test_index] = lgbm.predict(X_train[test_index], num_iteration=lgbm.best_iteration)
        xx_logloss.append(lgbm.best_score['valid']['binary_logloss'])
        print(xx_logloss)
    a = np.mean(xx_logloss)
    a = round(a, 5)
    print(a)

    sub = sub.sort_values(by='label', ascending=False)
    sub = sub.head(50000)
    sub = sub[['user_id', 'cate', 'shop_id', 'label']]
    sub.to_csv('./res/sub_F12_7.csv', index=False, index_label=False)



