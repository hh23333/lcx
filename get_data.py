# from datetime import datetime
# from datetime import timedelta
# from dateutil.relativedelta import relativedelta
import pandas as pd
import pickle
import os
import numpy as np
# from dateutil.parser import parse
import warnings
from sklearn import preprocessing
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')


def onehot(data, bin1,bin2):
    '''
    用于按步长生成数值型字段的one-hot
    :param data:需要进行one-hot编码的列
    :param bin1:步长左端值
    :param bin2:步长右端值
    :return: data的one-hot编码
    '''
    cate = ['{}-{}'.format(x,y) for (x,y) in zip(bin1,bin2)]
    one_hot = [[1 if (t>=x)&(t<y) else 0 for (x,y) in zip(bin1,bin2)] for t in data]
    df = pd.DataFrame(index=data.index,columns=cate,data=one_hot)
    return df


def fill_mode(data,features):
    '''
    用于用众数补充nan值
    :param data:需要填充nan值的dataframe
    :param features:需要填充nan值的列名
    :return: 众数填充完的data
    '''
    features_mode = {x:data[x].dropna().mode().values[0] for x in features}
    data.fillna(features_mode,inplace=True)
    return data


def fill_mode_group(data, features, keys=None):
    if keys is None:
        keys = '客户编号'
    values = data.groupby(keys).apply(lambda x:x[features].mode()).reset_index()
    values = dict(zip(values[keys], values[0]))
    data = data.set_index(keys)
    data[features].fillna(value=values, inplace=True)
    data = fill_mode(data.reset_index(), [features])
    return data


def fill_mean(data):
    '''
    用于用均值补充nan值
    :param data:需要填充nan值的列
    :return: 均值填充完的data
    '''
    data.fillna(data.mean(), inplace=True)  # 填充均值
    return data


def cate(profile_data, features, m):
    '''
    用于将数据进行聚类，并按客户编号groupby来填充nan值
    :param profile_data: profile数据(train和test一起进行聚类)
    :param features: 需要进行聚类的列,存储的值为数值型的列
    :param m: 聚类的个数
    :return: 聚类后的结果
    '''
    newname = '_'.join(features+['类别'])
    x = profile_data.set_index(['客户编号', '开户日期', '统计日期'])[features].replace('NA', np.nan).dropna()
    y = np.array(x).reshape(-1, 1)
    km = KMeans(n_clusters=m)
    km.fit(y)
    km.predict(y)
    z = pd.DataFrame(km.labels_, index=x.index, columns=[newname]).reset_index()
    profile_data = pd.merge(profile_data, z, on=['客户编号', '开户日期', '统计日期'],how='outer')
    profile_data_cate = fill_mode_group(profile_data, newname)
    return profile_data_cate


def clean_log_data(train_log_data, test_log_data):
    '''
    清理log数据： 去除数据中多余的空格，去除非客户编号及统计日期列的nan值
    :param train_log_data:
    :param test_log_data:
    :return:
    '''
    row = train_log_data.shape[0]
    log_data = pd.concat([train_log_data, test_log_data], join='inner')
    log_data.applymap((lambda x: "".join(x.split()) if type(x) is str else x))
    fillna = log_data.iloc[:, ~log_data.columns.isin(['客户编号', '统计日期'])].fillna(0)
    log_data = pd.concat([log_data.iloc[:, :2], fillna], axis=1)
    train_log_data = log_data.iloc[:row, :]
    test_log_data = log_data.iloc[row:, :]
    test_log_data = test_log_data.fillna({'统计日期': '2016-03'})

    return train_log_data, test_log_data


def clean_profile_data(train_profile_data, test_profile_data):
    '''
    清理数据：
            1. 返回的profile和label数据需要对齐，训练部分可以去除一些行，测试部分返回数据必须和原始数据有相同条数
            2. 一些column如年龄，开户时间等，最好多返回一列区间的值，如把9046,78,178，... --> 1,2,3区间编号，方便后续处理
            3. ...
    :param train_profile_data:
    :param test_profile_data:
    :return: 清理后的数据 train_profile_data, train_label, test_profile_data
    '''
    row = train_profile_data.shape[0]
    profile_data = pd.concat([train_profile_data, test_profile_data], join='inner')
    profile_data = profile_data.applymap((lambda x: "".join(x.split()) if type(x) is str else x))

    # 按年龄进行聚类并获取聚类的类别
    profile_data = cate(profile_data, ['年龄'], 5)
    # 按帐龄进行聚类并获取聚类的类别
    profile_data = cate(profile_data, ['帐龄'], 4)
    # 处理客户所在地区
    # key_mode = profile_data['客户所在地区'].mode().values[0]
    # profile_data['客户所在地区'] = profile_data['客户所在地区'].fillna(key_mode)
    profile_data = fill_mode_group(profile_data, '客户所在地区')
    # 处理性别
    profile_data = fill_mode_group(profile_data, '性别')
    # 处理新客户指数
    profile_data['新客户指数'].fillna(0, inplace=True)
    # 处理客户等级
    profile_data['客户等级'].fillna(1, inplace=True)
    # 处理最近活跃日期 基本无用 极少数有值
    # profile_data['最近活跃日期'].fillna('2015-01', inplace=True)
    # 处理客户关系类型
    profile_data = fill_mode_group(profile_data, '客户关系类型')
    # 处理客户类型
    profile_data['客户类型']=profile_data['客户类型'].apply(pd.to_numeric, errors='ignore')
    profile_data = fill_mode_group(profile_data, '客户类型')
    # 处理当前居住地区信息
    profile_data = fill_mode_group(profile_data, '当前居住地区信息')
    # 处理出生地信息
    profile_data = fill_mode_group(profile_data, '出生地信息')
    # 处理配偶指数 基本无用只有180个N 其他为nan
    # profile_data = fill_mode_group(profile_data, '配偶指数')
    # 客户加入渠道
    profile_data = fill_mode_group(profile_data, '客户加入的渠道')
    # 内部评价指数 只有1000个'S'
    profile_data = fill_mode_group(profile_data, '内部评价指数')
    # 地区名称 转为概率试试？
    profile_data = fill_mode_group(profile_data, '地区名称')
    # 客户分段
    profile_data = fill_mode_group(profile_data, '客户分段')
    # 员工标识
    profile_data = fill_mode_group(profile_data, '员工标识')
    profile_data['活跃度指标'].fillna(0, inplace=True)
    # 
    profile_data.drop(labels=['地址类型', '地区代码', '配偶指数', '最近活跃日期', '地址类型'],
                      axis=1, inplace=True)

    # 按原有数据切分train和test
    train_profile_data = profile_data.iloc[:row, :]
    test_profile_data = profile_data.iloc[row:, :]

    return train_profile_data, test_profile_data


def get_test_raw_data():
    test_profile_data = pd.read_pickle('./cache/test_profile_data.pkl')
    test_log_data = pd.read_pickle('./cache/test_log_data.pkl')
    anchor = test_profile_data[['客户编号', '开户日期']]
    return test_profile_data, test_log_data, anchor


def get_raw_data_bydate(train_start_date, train_end_date):

    '''
    :param label_date:
    :param train_start_date:
    :param train_end_date:
    :return:
    '''

    profile_data = pd.read_pickle('./cache/train_profile_data.pkl')
    log_data = pd.read_pickle('./cache/train_log_data.pkl')
    label_data = pd.read_pickle('./cache/train_label.pkl')

    profile_with_label = pd.merge(label_data['标签'], profile_data, how='outer', right_index=True, left_index=True)
    profile_with_label = profile_with_label[(profile_with_label['统计日期'] >= train_start_date)
                                            & (profile_with_label['统计日期'] <= train_end_date)]
    # log_end_date = datetime.strptime(train_start_date, '%Y-%m') - relativedelta(months=-1)
    # log_end_date = log_end_date.strftime('%Y-%m')
    log_data = log_data[(log_data['统计日期'] >= train_start_date)
                        & (log_data['统计日期'] < train_end_date)]

    return profile_with_label, log_data


def get_labels(label_date, raw_data_with_label):
    dump_path = './cache/train_labels_%s.pkl' % label_date
    if os.path.exists(dump_path):
        labels = pd.read_pickle(dump_path)
    else:
        labels = raw_data_with_label[raw_data_with_label['统计日期'] == label_date]
        # labels = labels[labels['标签'] == 1]
        labels = labels[['标签', '客户编号', '开户日期']]
        labels.to_pickle(dump_path)
    return labels


def get_basic_client_account_feat(raw_data, train_end_date, set_label='train'):
    '''
    将profile每一关键词转为特征
    :param raw_data:
    :param train_end_data:
    :return:
    '''
    dump_path = './cache/%s_basic_client_profile_%s.pkl' % (set_label, train_end_date)
    if os.path.exists(dump_path):
        client = pd.read_pickle(dump_path)
    else:
        # client_origin = pd.read_pickle('./cache/train_profile_data.pkl')
        client_origin = raw_data.drop_duplicates(subset=['客户编号', '开户日期'], keep='last')
        age_df = pd.get_dummies(client_origin['年龄_类别'], prefix='age')
        gender_df = pd.get_dummies(client_origin['性别'], prefix='gender')
        new_client_index_df = pd.get_dummies(client_origin['新客户指数'], prefix='new_client_index')
        account_age_df = pd.get_dummies(client_origin['帐龄_类别'], prefix='account_age')
        client_level_df = pd.get_dummies(client_origin['客户等级'], prefix='client_level')
        client_cat_df = pd.get_dummies(client_origin['客户类型'], prefix='client_cat')
        curr_res_df = pd.get_dummies(client_origin['当前居住地区信息'], prefix='curr_res_df')
        birthplace_df = pd.get_dummies(client_origin['出生地信息'], prefix='birthplace')
        client_rel_df = pd.get_dummies(client_origin['客户关系类型'], prefix='client_rel')
        # 加入渠道类型较多，使用该类型中，成交比例作为特征
        group_join = raw_data[raw_data['统计日期'] < train_end_date].groupby('客户加入的渠道')
        map_ = group_join['标签'].sum()/group_join.size()
        client_join_channel_df = client_origin['客户加入的渠道'].map(map_)
        inner_eval_index_df = pd.get_dummies(client_origin['内部评价指数'], prefix='inner_eval_index')
        # 地区类型较多，使用该类型中，成交比例作为特征
        group_area_name = raw_data[raw_data['统计日期'] < train_end_date].groupby('地区名称')
        map_ = group_area_name['标签'].sum() / group_area_name.size()
        area_name_df = client_origin['地区名称'].map(map_)
        client_seg_df = pd.get_dummies(client_origin['客户分段'], prefix='client_seg')
        staff_mark_df = pd.get_dummies(client_origin['员工标识'].replace('S', 'A'), prefix='staff_mark')
        activity_index_df = pd.get_dummies(client_origin['活跃度指标'], prefix='staff_mark')
        # 按客户编号和开户日期， merge 得到多维特征
        client = pd.concat([client_origin[['客户编号', '开户日期']],
                            age_df,
                            gender_df,
                            new_client_index_df,
                            account_age_df,
                            client_level_df,
                            client_cat_df,
                            curr_res_df,
                            birthplace_df,
                            client_rel_df,
                            client_join_channel_df,
                            inner_eval_index_df,
                            area_name_df,
                            client_seg_df,
                            staff_mark_df,
                            activity_index_df,
                            ], axis=1)

        pickle.dump(client, open(dump_path, 'wb'))
    print('client_account finished')
    return client


def get_history_client_count_feat(train_end_date, raw_date):
    pass


def get_client_extra_feat(profile_with_label, train_end_date, set_label='train'):
    '''
    用于获取客户的其他特征：是否为新客户，购买频率
    param profile_with_label:
    param train_end_date:训练截止日
    '''
    dump_path = './cache/%s_client_extra_F_%s.pkl' % (set_label, train_end_date)
    if os.path.exists(dump_path):
        client_extra_feat = pd.read_pickle(dump_path)
    else:
        client = profile_with_label['客户编号'].drop_duplicates()
        rate = pd.DataFrame(index=client)

        # 获取客户是否为新客户
        old = profile_with_label[profile_with_label['统计日期'] < train_end_date]['客户编号'].drop_duplicates()
        zero = pd.DataFrame(data=0, index=old, columns=['是否为新客户'])
        client_extra_feat = pd.merge(client, zero.reset_index(), on='客户编号', how='outer')
        client_extra_feat['是否为新客户'] = client_extra_feat['是否为新客户'].fillna(1)

        # 获取客户购买频率
        train_label = profile_with_label[profile_with_label['统计日期'] < train_end_date]
        rate['总次数'] = train_label.groupby('客户编号')['标签'].count()
        rate['购买信贷类产品次数'] = train_label[train_label['标签'] == 1].groupby('客户编号')['标签'].count()
        rate['购买信贷类产品频率'] = rate['购买信贷类产品次数'] / rate['总次数']
        rate['购买信贷类产品频率'] = rate['购买信贷类产品频率'].fillna(0)

        # 按label中的客户号来合并客户的特征
        client_extra_feat = pd.merge(client_extra_feat, rate['购买信贷类产品频率'].reset_index(), on='客户编号')
        pickle.dump(client_extra_feat, open(dump_path, 'wb'))
    return client_extra_feat


def get_log_feat(train_log_data, train_end_date, set_label='train'):
    dump_path = './cache/%s_log_F_%s.pkl' % (set_label, train_end_date)
    if os.path.exists(dump_path):
        train_log_feat = pd.read_pickle(dump_path)
    else:
        columns = list(train_log_data.iloc[:, 2:-1].columns)
        train_log_data = train_log_data.sort_values(['客户编号', '统计日期'])
        train_log_feat = pd.DataFrame(index=train_log_data['客户编号'].drop_duplicates(), columns=columns, data=0)
        for col in columns:
            x = train_log_data.groupby(['客户编号', col]).tail(1)
            y = x.pivot(index='客户编号', columns=col, values='统计日期').loc[:, 1].fillna('2015-01')
            train_log_feat[col] = y.apply(lambda x:(pd.to_datetime(train_end_date)-pd.to_datetime(x)).days//30)
            train_log_feat.rename(columns={col: ''.join(['距离最近一次']+[col.replace('是否', '')]+['的月数'])})
        train_log_feat = train_log_feat.fillna(14)
        pickle.dump(train_log_feat, open(dump_path, 'wb'))
    return train_log_feat


def get_train_data(label_date, train_start_date, train_end_date, ):
    dump_path = './cache/train_set_F_%s_%s_%s.pkl' % (
        train_start_date, train_end_date, label_date)
    if os.path.exists(dump_path):
        labels = pd.read_pickle(dump_path)
    else:
        raw_data_with_label, log_data = get_raw_data_bydate(train_start_date, train_end_date)
        # raw_data_with_label = raw_data_with_label.drop_duplicates(['客户编号', '统计日期', '开户日期'], keep='last')

        labels = get_labels(label_date, raw_data_with_label)
        pos_labels = labels[labels['标签'] == 1]
        neg_labels = labels[labels['标签'] == 0]

        # 特征
        # 统计用户账户的基本特征，以['客户编号', '开户日期'] group
        client_count_basic_feat = get_basic_client_account_feat(raw_data_with_label, train_end_date)
        # 以['客户编号'] group 统计用户账户额外的特征
        client_extra_feat = get_client_extra_feat(raw_data_with_label, train_end_date)
        # 可使用历史label的特征], 以['客户编号', '开户日期'] group
        # client_count_history_feat = get_history_client_count_feat(label_date, raw_data_with_label)
        # 对log提特征， 以['客户编号'] group
        client_log_feat = get_log_feat(log_data, train_end_date)

        # 负样本采样, 负样本为正样本的8倍
        frac = (pos_labels.shape[0] * 6) / neg_labels.shape[0]
        neg_labels = neg_labels.sample(frac=frac, replace=True).reset_index(drop=True)

        labels = pd.concat([pos_labels, neg_labels], axis=0, ignore_index=True)

        labels = pd.merge(labels, client_count_basic_feat, how='left', on=['客户编号', '开户日期'])
        # labels = pd.merge(labels, client_count_history_feat, how='left', on=['客户编号', '开户日期'])
        labels = pd.merge(labels, client_extra_feat, how='left', on=['客户编号'])
        labels = pd.merge(labels, client_log_feat, how='left', on=['客户编号'])

        # TODO 更改填充方式，使用均值会不会好一点
        labels.fillna(0)
        labels.to_pickle(dump_path)

    return labels


def get_test_data(test_date, train_start_date, train_end_date):
    dump_path = './cache/test_set_F_%s_%s_%s.pkl' % (
        train_start_date, train_end_date, test_date)
    if os.path.exists(dump_path):
        test_anchor = pd.read_pickle(dump_path)
    else:
        raw_data_with_label, log_data = get_raw_data_bydate(train_start_date, train_end_date)
        # raw_data_with_label = raw_data_with_label.drop_duplicates(['客户编号', '统计日期', '开户日期'], keep='last')

        test_raw_profile_data, test_log_data, test_anchor = get_test_raw_data()
        all_raw_profile_data = pd.concat([raw_data_with_label, test_raw_profile_data], axis=0)
        all_raw_log_data = pd.concat([log_data, test_log_data], axis=0)

        # 特征
        # 统计用户账户的基本特征，以['客户编号', '开户日期'] group
        client_count_basic_feat = get_basic_client_account_feat(all_raw_profile_data, test_date, 'test')
        # 以['客户编号'] group 统计用户账户额外的特征
        client_extra_feat = get_client_extra_feat(all_raw_profile_data, test_date, 'test')
        # 可使用历史label的特征], 以['客户编号', '开户日期'] group
        # client_count_history_feat = get_history_client_count_feat(test_date, all_raw_profile_data)
        # 对log提特征， 以['客户编号'] group
        client_log_feat = get_log_feat(all_raw_log_data, test_date, 'test')

        test_anchor = pd.merge(test_anchor, client_count_basic_feat, how='left', on=['客户编号', '开户日期'])
        # test_anchor = pd.merge(test_anchor, client_count_history_feat, how='left', on=['客户编号', '开户日期'])
        test_anchor = pd.merge(test_anchor, client_extra_feat, how='left', on=['客户编号'])
        test_anchor = pd.merge(test_anchor, client_log_feat, how='left', on=['客户编号'])

        # TODO 更改填充方式，使用均值会不会好一点
        test_anchor.fillna(0)
        test_anchor.to_pickle(dump_path)

    return test_anchor

