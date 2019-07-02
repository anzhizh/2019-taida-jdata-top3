from user_cate_shop2 import *


# 读取行为数据，与产品数据拼接（用于生成购物车特征）
def get_actions_product_cart(start_date, end_date):
    dump_path = './cache/all_action_product_cart_F11_5_%s_%s.pkl' % (start_date, end_date)
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
        #actions.to_pickle(dump_path)
    return actions


# 行为比例特征（2.01-4.08） 滑窗
def get_accumulate_user_feat_v1(start_date, end_date):
    dump_path = './cache/user_feat_v1_accumulate_F11_5_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        f11_actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)

        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)

        # 索引
        f11_actions = actions[['user_id', 'cate']].drop_duplicates()

        actions1 = actions.drop(['cate'], axis=1)
        actions1 = actions1.groupby(['user_id'], as_index=False).sum().add_prefix('user_id_')
        actions1['user_action_1_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (start_date, end_date)] / actions1['user_id_%s-%s-action_1' % (start_date, end_date)]
        actions1['user_action_4_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (start_date, end_date)] / actions1['user_id_%s-%s-action_4' % (start_date, end_date)]
        actions1['user_action_3_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (start_date, end_date)] / actions1['user_id_%s-%s-action_3' % (start_date, end_date)]
        actions1.rename(columns={'user_id_user_id': 'user_id'}, inplace=True)

        actions2 = actions.drop(['user_id'], axis=1)
        actions2 = actions2.groupby(['cate'], as_index=False).sum().add_prefix('cate_')
        actions2['cate_action_1_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (start_date, end_date)] / actions2['cate_%s-%s-action_1' % (start_date, end_date)]
        actions2['cate_action_4_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (start_date, end_date)] / actions2['cate_%s-%s-action_4' % (start_date, end_date)]
        actions2['cate_action_3_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (start_date, end_date)] / actions2['cate_%s-%s-action_3' % (start_date, end_date)]
        actions2.rename(columns={'cate_cate': 'cate'}, inplace=True)

        actions4 = actions
        actions4 = actions4.groupby(['user_id', 'cate'], as_index=False).sum().add_prefix('user_cate_shop_id_')
        actions4['user_cate_shop_id_action_1_ratio_%s_%s' % (start_date, end_date)] = actions4['user_cate_shop_id_%s-%s-action_2' % (start_date, end_date)] / actions4['user_cate_shop_id_%s-%s-action_1' % (start_date, end_date)]
        actions4['user_cate_shop_id_action_4_ratio_%s_%s' % (start_date, end_date)] = actions4['user_cate_shop_id_%s-%s-action_2' % (start_date, end_date)] / actions4['user_cate_shop_id_%s-%s-action_4' % (start_date, end_date)]
        actions4['user_cate_shop_id_action_3_ratio_%s_%s' % (start_date, end_date)] = actions4['user_cate_shop_id_%s-%s-action_2' % (start_date, end_date)] / actions4['user_cate_shop_id_%s-%s-action_3' % (start_date, end_date)]
        actions4.rename(columns={'user_cate_shop_id_user_id': 'user_id', 'user_cate_shop_id_cate': 'cate'}, inplace=True)

        # 拼接
        f11_actions = f11_actions.merge(actions1, on='user_id', how='left')
        f11_actions = f11_actions.merge(actions2, on='cate', how='left')
        f11_actions = f11_actions.merge(actions4, on=['user_id', 'cate'], how='left')
        #f11_actions.to_pickle(dump_path)
    print('accumulate user finished')
    return f11_actions


def get_accumulate_user_cart_feat_v1(start_date, end_date):
    dump_path = './cache/user_cart_feat_v1_accumulate_F11_5_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        f11_actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product_cart(start_date, end_date)

        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)

        # 索引
        f11_actions = actions[['user_id', 'cate']].drop_duplicates()

        actions1 = actions.drop(['cate'], axis=1)
        actions1 = actions1.groupby(['user_id'], as_index=False).sum().add_prefix('user_id_')
        actions1['user_action_1_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (start_date, end_date)] / actions1['user_id_%s-%s-action_1' % (start_date, end_date)]
        actions1['user_action_4_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (start_date, end_date)] / actions1['user_id_%s-%s-action_4' % (start_date, end_date)]
        actions1['user_action_3_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (start_date, end_date)] / actions1['user_id_%s-%s-action_3' % (start_date, end_date)]
        actions1['user_action_5_ratio_%s_%s' % (start_date, end_date)] = actions1['user_id_%s-%s-action_2' % (
        start_date, end_date)] / actions1['user_id_%s-%s-action_5' % (start_date, end_date)]

        actions1.rename(columns={'user_id_user_id': 'user_id'}, inplace=True)

        actions2 = actions.drop(['user_id'], axis=1)
        actions2 = actions2.groupby(['cate'], as_index=False).sum().add_prefix('cate_')
        actions2['cate_action_1_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (start_date, end_date)] / actions2['cate_%s-%s-action_1' % (start_date, end_date)]
        actions2['cate_action_4_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (start_date, end_date)] / actions2['cate_%s-%s-action_4' % (start_date, end_date)]
        actions2['cate_action_3_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (start_date, end_date)] / actions2['cate_%s-%s-action_3' % (start_date, end_date)]
        actions2['cate_action_5_ratio_%s_%s' % (start_date, end_date)] = actions2['cate_%s-%s-action_2' % (
        start_date, end_date)] / actions2['cate_%s-%s-action_5' % (start_date, end_date)]

        actions2.rename(columns={'cate_cate': 'cate'}, inplace=True)

        actions4 = actions
        actions4 = actions4.groupby(['user_id', 'cate'], as_index=False).sum().add_prefix('user_cate_shop_id_')
        actions4['user_cate_shop_id_action_1_ratio_%s_%s' % (start_date, end_date)] = actions4['user_cate_shop_id_%s-%s-action_2' % (start_date, end_date)] / actions4['user_cate_shop_id_%s-%s-action_1' % (start_date, end_date)]
        actions4['user_cate_shop_id_action_4_ratio_%s_%s' % (start_date, end_date)] = actions4['user_cate_shop_id_%s-%s-action_2' % (start_date, end_date)] / actions4['user_cate_shop_id_%s-%s-action_4' % (start_date, end_date)]
        actions4['user_cate_shop_id_action_3_ratio_%s_%s' % (start_date, end_date)] = actions4['user_cate_shop_id_%s-%s-action_2' % (start_date, end_date)] / actions4['user_cate_shop_id_%s-%s-action_3' % (start_date, end_date)]
        actions4['user_cate_shop_id_action_5_ratio_%s_%s' % (start_date, end_date)] = actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_2' % (
                                                                                          start_date, end_date)] / \
                                                                                      actions4[
                                                                                          'user_cate_shop_id_%s-%s-action_5' % (
                                                                                          start_date, end_date)]

        actions4.rename(columns={'user_cate_shop_id_user_id': 'user_id', 'user_cate_shop_id_cate': 'cate'}, inplace=True)

        # 拼接
        f11_actions = f11_actions.merge(actions1, on='user_id', how='left')
        f11_actions = f11_actions.merge(actions2, on='cate', how='left')
        f11_actions = f11_actions.merge(actions4, on=['user_id', 'cate'], how='left')
        #f11_actions.to_pickle(dump_path)

    print('accumulate user cart finished')
    return f11_actions


# 基础统计特征
def get_stat_feat_v1(start_date, end_date):
    dump_path = './cache/stat_feat_accumulate_v1_F11_5_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        action = pd.read_pickle(dump_path)
    else:
        action = get_actions_product(start_date, end_date)
        action_index = action[['user_id', 'cate']].drop_duplicates()

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
        cate_order_user_count = cate_order_user_count[cate_order_user_count.act_2 > 0].groupby('cate')['user_id'].nunique()
        cate_order_user_rate = (cate_order_user_count / cate_user_nunique)
        cate_sku_nunique = action.groupby('cate')['sku_id'].nunique()

        # cate下的店铺特征
        cate_shop_count = action.groupby('cate')['shop_id'].count()
        cate_shop_nunique = action.groupby('cate')['shop_id'].nunique()
        cate_shop_order_count = action_type.groupby('cate')['act_2'].sum()
        cate_shop_order_rate = cate_shop_order_count / cate_shop_count

        # cate下： 购买店铺/总店铺
        cate_order_shop_count = action_type.groupby(['cate', 'shop_id'])['act_2'].sum().reset_index()
        cate_order_shop_count = cate_order_shop_count[cate_order_shop_count.act_2 > 0].groupby('cate')['shop_id'].nunique()
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

        action = pd.merge(action_index, user_stat, on='user_id', how='left')
        action = pd.merge(action, cate_stat, on='cate', how='left')
        #action.to_pickle(dump_path)
    print('stat_feat finished')
    return action


# 交叉特征
def get_cross_feat_v1(start_date, end_date):
    dump_path = './cache/cross_feat_v1_F11_5_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)[['user_id', 'cate']]
        actions['cnt'] = 0

        action1 = actions.groupby(['user_id', 'cate'], as_index=False).count()

        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['cate']
        action2.columns = ['user_id', 'user_cnt']

        action3 = actions.groupby('cate', as_index=False).count()
        del action3['user_id']
        action3.columns = ['cate', 'cate_cnt']
        actions = pd.merge(action1, action2, how='left', on='user_id')
        actions = pd.merge(actions, action3, how='left', on='cate')

        actions['user_cnt'] = actions['cnt'] / actions['user_cnt']
        actions['cate_cnt'] = actions['cnt'] / actions['cate_cnt']
        del actions['cnt']
        #pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id', 'cate'] + ['cross_feat_' + str(i) for i in range(1, actions.shape[1] - 1)]
    print('cross feature finished')
    return actions


# U_B对行为1，2，4，5进行 浏览次数/用户总浏览次数（或者物品的浏览次数）
def get_user_feat15_v1(start_date, end_date):
    dump_path = './cache/user_feat15_v1_F11_5_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
        actions.columns = ['user_id', 'cate'] + ['user_feat15_' + str(i) for i in
                                                            range(1, actions.shape[1] - 1)]
        return actions
    else:
        temp = None
        df = get_actions_product(start_date, end_date)[['user_id', 'cate', 'type']]
        for i in (1, 2, 3):
            actions = df[df['type'] == i]
            action1 = actions.groupby(['user_id', 'cate'], as_index=False).count()
            action1.columns = ['user_id', 'cate', 'visit']

            action2 = actions.groupby('user_id', as_index=False).count()
            del action2['type']
            action2.columns = ['user_id', 'user_visits_cate']

            action4 = actions.groupby('cate', as_index=False).count()
            del action4['type']
            action4.columns = ['cate', 'cate_visits_user']

            actions = pd.merge(action1, action2, how='left', on='user_id')
            actions = pd.merge(actions, action4, how='left', on='cate')

            actions['visit_rate_user1'] = actions['visit'] / actions['user_visits_cate']
            actions['visit_rate_cate1'] = actions['visit'] / actions['cate_visits_user']
            if temp is None:
                temp = actions
            else:
                temp = pd.merge(temp, actions, how="outer", on=['user_id', 'cate'])
        #pickle.dump(temp, open(dump_path, 'wb'))
        temp.columns = ['user_id', 'cate'] + ['user_feat15_' + str(i) for i in
                                                            range(1, temp.shape[1] - 1)]
        return temp


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

    x_act5_stat['cart_minus_buy'] = x_act5_stat['lastday_sum_act_5'] - x_act5_stat['lastday_sum_act_2']

    return x_act5_stat


# 标签
def get_labels_v1(start_date, end_date):
    dump_path = './cache/labels_v1_F11_5_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions_product(start_date, end_date)
        actions = actions[actions['type'] == 2]
        actions = actions.groupby(['user_id', 'cate'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'cate', 'label']]
        #actions.to_pickle(dump_path)
    print('label finished')
    return actions


def make_train_set_F11_5(train_start_date, train_end_date, test_start_date, test_end_date, start):
    dump_path = './cache/train_set_v1_F11_5_%s_%s_%s_%s.pkl' % (
        train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 索引
        f11_actions = get_actions_product(train_start_date, train_end_date)
        f11_actions = f11_actions.drop_duplicates(['user_id', 'cate'])
        f11_actions = f11_actions[['user_id', 'cate']]

        # 标签
        labels = get_labels(test_start_date, test_end_date)

        # 特征
        start_days = "2018-02-01"                              #
        user = get_basic_user_feat()
        product_stat = get_product_stat_feat(start_days, train_end_date)
        time = get_time_feat(start_days, train_end_date)
        stat_feat = get_stat_feat_v1(start_days, train_end_date)
        user_feat = user_features(start_days, train_end_date)
        cross_feat = get_cross_feat_v1(start_days, train_end_date)

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
        user_feat15 = get_user_feat15_v1(start_days, train_end_date)           #

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

        # 滑窗行为特征
        actions = None
        for i in (5, 7, 14, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_accumulate_user_feat_v1(start_days, train_end_date)
            else:
                actions1 = get_accumulate_user_feat_v1(start_days, train_end_date)
                actions = pd.merge(actions, actions1, how='left', on=['user_id', 'cate'])

        # 前3天滑窗行为 包含cart
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=3)
        start_days = start_days.strftime('%Y-%m-%d')
        actions_cart = get_accumulate_user_cart_feat_v1(start_days, train_end_date)

        # act_5
        act5_feat = get_last1day_cart_fearture(start_days, train_end_date, 3)
        act5_feat = act5_feat.groupby(['user_id', 'cate'], as_index=False).sum()
        del act5_feat['shop_id']

        # 负采样
        f11_actions = pd.merge(f11_actions, labels, how='left', on=['user_id', 'cate'])
        f11_actions = f11_actions.fillna(0)
        print('train data size:', f11_actions.shape[0])
        f11_actions_1 = f11_actions[f11_actions['label'] == 1]
        f11_actions_0 = f11_actions[f11_actions['label'] == 0]
        frac1 = (f11_actions_1.shape[0] * 30) / f11_actions_0.shape[0]  # 负样本为正样本30倍
        f11_actions_0 = f11_actions_0.sample(frac=frac1).reset_index(drop=True)
        f11_actions = pd.concat([f11_actions_1, f11_actions_0], axis=0, ignore_index=True)
        f11_actions = f11_actions.sample(frac=1).reset_index(drop=True)
        print('train data size after sample:', f11_actions.shape[0])

        actions = pd.merge(f11_actions, actions, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, actions_cart, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, time, how='left', on='user_id')
        actions = pd.merge(actions, stat_feat, how='left', on=['user_id', 'cate'])
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
        actions = pd.merge(actions, user_feat15, how='left', on=['user_id', 'cate'])

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

        actions = pd.merge(actions, act5_feat, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, cross_feat, how='left', on=['user_id', 'cate'])
        actions = actions.fillna(0)
        # actions.to_pickle(dump_path)
    print('train_set finised')
    return actions


def make_test_set_F11_5(train_start_date, train_end_date,start):
    dump_path = './cache/test_set_F11_5_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        # 索引
        f11_actions = get_actions_product(train_start_date, train_end_date)
        f11_actions = f11_actions.drop_duplicates(['user_id', 'cate'])
        f11_actions = f11_actions[['user_id', 'cate']]  #

        # 特征
        start_days = "2018-02-01"  #
        user = get_basic_user_feat()
        product_stat = get_product_stat_feat(start_days, train_end_date)
        time = get_time_feat(start_days, train_end_date)
        stat_feat = get_stat_feat_v1(start_days, train_end_date)
        user_feat = user_features(start_days, train_end_date)
        cross_feat = get_cross_feat_v1(start_days, train_end_date)

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
        user_feat15 = get_user_feat15_v1(start_days, train_end_date)  #

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

        # 滑窗行为特征
        actions = None
        for i in (5, 7, 14, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_accumulate_user_feat_v1(start_days, train_end_date)
            else:
                actions1 = get_accumulate_user_feat_v1(start_days, train_end_date)
                actions = pd.merge(actions, actions1, how='left', on=['user_id', 'cate'])

        # 前3天滑窗行为 包含cart
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=3)
        start_days = start_days.strftime('%Y-%m-%d')
        actions_cart = get_accumulate_user_cart_feat_v1(start_days, train_end_date)

        # act_5
        act5_feat = get_last1day_cart_fearture(start_days, train_end_date, 3)
        act5_feat = act5_feat.groupby(['user_id', 'cate'], as_index=False).sum()
        del act5_feat['shop_id']

        actions = pd.merge(f11_actions, actions, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, actions_cart, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, time, how='left', on='user_id')
        actions = pd.merge(actions, stat_feat, how='left', on=['user_id', 'cate'])
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
        actions = pd.merge(actions, user_feat15, how='left', on=['user_id', 'cate'])

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

        actions = pd.merge(actions, act5_feat, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, cross_feat, how='left', on=['user_id', 'cate'])
        actions = actions.fillna(0)
        del stat_feat, f11_actions
    print('test_set finished')
    return actions


def lgb_train_F11_5(X_train1, y_train1, X_test1, sub_user_index):
    # 提交结果
    sub = sub_user_index[['user_id', 'cate']].copy()
    sub['shop_id'] = 0
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
        'nthread': -1,  # -1
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

    sub.to_csv('./res/sub_F11_5.csv', index=False, index_label=False)