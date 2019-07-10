###########################common##########################################
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, Rating
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json
import sys
import time
import csv
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
sc_conf = SparkConf().setAppName('inf553_project').setMaster('local[*]').set('spark.driver.memory', '4g')\
    .set('spark.executor.memory', '4g').set('spark.driver.maxResultSize', '4g')
sc = SparkContext(conf=sc_conf)
file_path = sys.argv[1]
input_file = file_path + 'yelp_train.csv'
user_file = file_path + 'user.json'
business_file = file_path + 'business.json'
test_file = sys.argv[2]
output_file = sys.argv[3]
train_rdd = sc.textFile(input_file)
header = 'user_id, business_id, stars'
train_data = train_rdd.filter(lambda s: s != header).map(lambda s:s.split(',')).persist()
test_data = sc.textFile(test_file).filter(lambda s: s != header).map(lambda s:s.split(',')).persist()
user_id = train_data.map(lambda s: s[0]).distinct().sortBy(lambda s:s).collect()
business_id = train_data.map(lambda s: s[1]).distinct().sortBy(lambda s:s).collect()
user_based_data = train_data.map(lambda s: (s[0], (s[1], s[2]))).groupByKey().sortByKey().map(lambda s:(s[0],list(s[1])))
all_user_dic = {}
def to_dic(s):
    one_user_dic = {}
    one_busi_dic = {}
    for item in s[1]:
        one_busi_dic[item[0]] = float(item[1])
    one_user_dic[s[0]] = one_busi_dic
    return one_user_dic


user_dic = user_based_data.map(to_dic).collect()
for a_user_dic in user_dic:
    for users in a_user_dic:
        all_user_dic[users] = a_user_dic[users]


def filter_new_item(s):
    new_list = []
    if s[1] not in business_id:
        new_item_score = sum(all_user_dic[s[0]].values()) / len(all_user_dic[s[0]])
        new_list.append((1, (s[0], s[1], new_item_score)))
    elif s[0] not in user_id:
        new_list.append((1, (s[0], s[1], 3)))
    else:
        new_list.append((0, s))
    return new_list


ff = test_data.flatMap(filter_new_item).groupByKey().map(lambda s: list(s[1])).collect()
filtered_test = ff[0]
final_predict = ff[1]
##############################linear_regresion###############################
user_file_data = sc.textFile(user_file).map(lambda s:(json.loads(s)['user_id'],json.loads(s)['review_count'],json.loads(s)['average_stars'])).collect()
business_file_data = sc.textFile(business_file).map(lambda s:(json.loads(s)['business_id'],json.loads(s)['stars'],json.loads(s)['review_count'])).collect()
user_info_dic = {}
for item in user_file_data:
    info = []
    info = info + [item[1]] + [item[2]]
    user_info_dic[item[0]] = info
busi_info_dic = {}
for item in business_file_data:
    info = []
    info = info + [item[1]] + [item[2]]
    busi_info_dic[item[0]] = info
train_data_get = train_data.collect()
test_data_get = test_data.collect()
new_train_data = []
for a_data in train_data_get:
    one_data = []
    one_data = one_data + user_info_dic[a_data[0]] + busi_info_dic[a_data[1]] + [float(a_data[2])]
    new_train_data.append(one_data)
new_train_data = pd.DataFrame(new_train_data)
new_test_data = []
for a_data in test_data_get:
    one_data = []
    one_data = one_data + user_info_dic[a_data[0]] + busi_info_dic[a_data[1]] + [float(a_data[2])]
    new_test_data.append(one_data)
new_test_data = pd.DataFrame(new_test_data)
new_train_only_data = new_train_data.iloc[:,0:4]
new_train_label = new_train_data.iloc[:,4]
new_test_only_data = new_test_data.iloc[:,0:4]
clf = LinearRegression().fit(new_train_only_data, new_train_label)
y_pre = clf.predict(new_test_only_data)
linear_prediction = []
for i in range(len(y_pre)):
    all_info = [test_data_get[i][0]] + [test_data_get[i][1]] + [y_pre[i]]
    linear_prediction.append(all_info)
####################################surprise######################################
surprise_reader = Reader(line_format='user item rating', sep=',', skip_lines= 1)
surprise_train = Dataset.load_from_file(input_file, reader=surprise_reader)
surprise_train = surprise_train.build_full_trainset()
surprise_test_data = sc.parallelize(test_data_get).map(lambda s:(s[0], s[1], float(s[2]))).collect()
params = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
surprise_formula = BaselineOnly(bsl_options=params)
surprise_formula.fit(surprise_train)
surprise_predict = surprise_formula.test(surprise_test_data)
surprise_prediction = []
for i in range(len(surprise_predict)):
    surprise_prediction.append([surprise_predict[i][0], surprise_predict[i][1], surprise_predict[i][3]])
################################SVD########################################
from surprise import SVD
svd_surprise = SVD(n_epochs=30, lr_all=0.008, reg_all=0.2)
svd_surprise.fit(surprise_train)
surprise_svd_prediction = svd_surprise.test(surprise_test_data)
svd_prediction = []
for i in range(len(surprise_svd_prediction)):
    svd_prediction.append([surprise_svd_prediction[i][0], surprise_svd_prediction[i][1], surprise_svd_prediction[i][3]])
############################item-based#####################################
rank = 7
need_predict_data = sc.parallelize(filtered_test).map(lambda s: (s[0], s[1])).groupByKey().map(lambda s: (s[0], list(s[1]))).collect()
invert_business_dic = {}
invert_business = train_data.map(lambda s:(s[1],s[0])).groupByKey().map(lambda s:(s[0],list(s[1]))).collect()
for business_invert in invert_business:
    invert_business_dic[business_invert[0]] = business_invert[1]


def item_predict_score_user(s):
    user = s[0]
    user_avg = sum(all_user_dic[user].values()) / len(all_user_dic[user])
    need_pre_user_item = all_user_dic[user]
    business_group = s[1]
    final_predict = []
    for a_business in business_group:
        pearson_score_dic = {}
        other_users_rate_this = invert_business_dic[a_business]
        for other_users in other_users_rate_this:
            co_rated_items = set(all_user_dic[other_users].keys()) & set(need_pre_user_item.keys())
            if len(co_rated_items) == 1:
                pearson_score_dic[other_users] = 0
            elif len(co_rated_items) == 0:
                continue
            else:
                candidate_user_items = []
                need_predict_user_items = []
                for co_item in co_rated_items:
                    candidate_user_items.append(all_user_dic[other_users][co_item])
                    need_predict_user_items.append(need_pre_user_item[co_item])

                candidate_avg = sum(candidate_user_items) / len(candidate_user_items)
                need_predict_avg = sum(need_predict_user_items) / len(need_predict_user_items)
                molecule = 0
                left_denominator = 0
                right_denominator = 0
                for i in range(len(candidate_user_items)):
                    molecule += ((need_predict_user_items[i] - need_predict_avg) * (
                                candidate_user_items[i] - candidate_avg))
                    left_denominator += ((need_predict_user_items[i] - need_predict_avg) ** 2)
                    right_denominator += ((candidate_user_items[i] - candidate_avg) ** 2)
                if (molecule == 0):
                    pearson_score_dic[other_users] = 0
                else:
                    pearson_score = molecule / ((left_denominator ** 0.5) * (right_denominator ** 0.5))
                    pearson_score_dic[other_users] = pearson_score
        L = sorted(pearson_score_dic.items(), key=lambda item: item[1], reverse=True)
        L = L[:rank]
        top_score_users = {}
        for l in L:
            top_score_users[l[0]] = l[1]
        sum_above = 0
        sum_below = 0
        for top_users in top_score_users:
            other_avg = (sum(all_user_dic[top_users].values()) - all_user_dic[top_users][a_business]) \
                        / (len(all_user_dic[top_users]) - 1)
            sum_above += ((all_user_dic[top_users][a_business] - other_avg) * float(top_score_users[top_users]))
            sum_below += abs(float(top_score_users[top_users]))
        if (sum_below == 0):
            predict_score = user_avg
        else:
            predict_score = user_avg + (sum_above / sum_below)
        final_predict.append((user, a_business, predict_score))
    return final_predict


item_predictions = sc.parallelize(need_predict_data).map(item_predict_score_user).collect()
item_need_predict_list = []
for rows in item_predictions:
    for row in rows:
        item_need_predict_list.append(row)
item_based_prediction = []
for rows in item_need_predict_list:
    if rows[2] <= 1:
        item_based_prediction.append((rows[0], rows[1], 1))
    elif rows[2] >= 5:
        item_based_prediction.append((rows[0], rows[1], 5))
    else:
        item_based_prediction.append((rows[0], rows[1], rows[2]))
for rows in final_predict:
    item_based_prediction.append(rows)
############################user-based#####################################
rank = 10
need_predict_data = sc.parallelize(filtered_test).map(lambda s: (s[0], s[1])).groupByKey().map(lambda s: (s[0], list(s[1]))).collect()
invert_business_dic = {}
invert_business = train_data.map(lambda s:(s[1],s[0])).groupByKey().map(lambda s:(s[0],list(s[1]))).collect()
for business_invert in invert_business:
    invert_business_dic[business_invert[0]] = business_invert[1]


def user_predict_score_user(s):
    user = s[0]
    user_avg = sum(all_user_dic[user].values()) / len(all_user_dic[user])
    need_pre_user_item = all_user_dic[user]
    business_group = s[1]
    final_predict = []
    for a_business in business_group:
        pearson_score_dic = {}
        other_users_rate_this = invert_business_dic[a_business]
        for other_users in other_users_rate_this:
            co_rated_items = set(all_user_dic[other_users].keys()) & set(need_pre_user_item.keys())
            if len(co_rated_items) == 1:
                pearson_score_dic[other_users] = 0
            elif len(co_rated_items) == 0:
                continue
            else:
                candidate_user_items = []
                need_predict_user_items = []
                for co_item in co_rated_items:
                    candidate_user_items.append(all_user_dic[other_users][co_item])
                    need_predict_user_items.append(need_pre_user_item[co_item])

                candidate_avg = sum(candidate_user_items) / len(candidate_user_items)
                need_predict_avg = sum(need_predict_user_items) / len(need_predict_user_items)
                molecule = 0
                left_denominator = 0
                right_denominator = 0
                for i in range(len(candidate_user_items)):
                    molecule += ((need_predict_user_items[i] - need_predict_avg) * (
                                candidate_user_items[i] - candidate_avg))
                    left_denominator += ((need_predict_user_items[i] - need_predict_avg) ** 2)
                    right_denominator += ((candidate_user_items[i] - candidate_avg) ** 2)
                if (molecule == 0):
                    pearson_score_dic[other_users] = 0
                else:
                    pearson_score = molecule / ((left_denominator ** 0.5) * (right_denominator ** 0.5))
                    pearson_score_dic[other_users] = pearson_score
        L = sorted(pearson_score_dic.items(), key=lambda item: item[1], reverse=True)
        L = L[:rank]
        top_score_users = {}
        for l in L:
            top_score_users[l[0]] = l[1]
        sum_above = 0
        sum_below = 0
        for top_users in top_score_users:
            other_avg = (sum(all_user_dic[top_users].values()) - all_user_dic[top_users][a_business]) \
                        / (len(all_user_dic[top_users]) - 1)
            sum_above += ((all_user_dic[top_users][a_business] - other_avg) * float(top_score_users[top_users]))
            sum_below += abs(float(top_score_users[top_users]))
        if (sum_below == 0):
            predict_score = user_avg
        else:
            predict_score = user_avg + (sum_above / sum_below)
        final_predict.append((user, a_business, predict_score))
    return final_predict


user_predictions = sc.parallelize(need_predict_data).map(user_predict_score_user).collect()
user_need_predict_list = []
for rows in user_predictions:
    for row in rows:
        user_need_predict_list.append(row)
user_based_prediction = []
for rows in user_need_predict_list:
    if rows[2] <= 1:
        user_based_prediction.append((rows[0], rows[1], 1))
    elif rows[2] >= 5:
        user_based_prediction.append((rows[0], rows[1], 5))
    else:
        user_based_prediction.append((rows[0], rows[1], rows[2]))
for rows in final_predict:
    user_based_prediction.append(rows)
coef = [ 0.34612386,  0.69090598,  0.08184975, -0.00815533,  0.02872788,-0.08471383,  0.02176268, -0.00522962]
intercept = -0.30974417790196007
avg_dic = {}
max_dic = {}
min_dic = {}
for user in all_user_dic:
    avg_dic[user] = sum(all_user_dic[user].values()) / len(all_user_dic[user])
    max_dic[user] = max(all_user_dic[user].values())
    min_dic[user] = min(all_user_dic[user].values())
surprise_dic = {}
linear_dic = {}
svd_dic = {}
item_dic = {}
user_dic = {}
#avg_dic  max_dic min_dic
for i in surprise_prediction:
    surprise_dic[(i[0],i[1])] = i[2]
for i in linear_prediction:
    linear_dic[(i[0],i[1])] = i[2]
for i in svd_prediction:
    svd_dic[(i[0],i[1])] = i[2]
for i in item_based_prediction:
    item_dic[(i[0],i[1])] = i[2]
for i in user_based_prediction:
    user_dic[(i[0],i[1])] = i[2]
final_final_prediction = []
for i in range(len(surprise_prediction)):
    user_info = surprise_prediction[i][0]
    business_info = surprise_prediction[i][1]
    u_b_key = tuple((user_info,business_info))
    final_score = coef[0] * surprise_dic[u_b_key] + coef[1] * linear_dic[u_b_key]+ coef[2] * svd_dic[u_b_key] + coef[3] * item_dic[u_b_key] \
        + coef[4] * user_dic[u_b_key] + coef[5] * avg_dic[user_info] + coef[6] * max_dic[user_info]\
        + coef[7] * min_dic[user_info] + intercept
    final_final_prediction += [[user_info]+[business_info]+[final_score]]
title = ['user_id','business_id','prediction']
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(title)
    for i in final_final_prediction:
        writer.writerow(i)
