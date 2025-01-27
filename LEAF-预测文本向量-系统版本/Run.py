import pdb

from main import *
import argparse

import eel
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

method = ""
section = ""
model = ""
prospects = ""
strategy = ""


@eel.expose
def save_method(*args):
    global method
    method = args[0]


@eel.expose
def save_section(*args):
    global section
    section = args[0]

@eel.expose
def save_model(*args):
    global model
    model = args[0]

@eel.expose
def save_prospects(*args):
    global prospects
    prospects = args[0]

@eel.expose
def save_strategy(*args):
    global strategy
    strategy = args[0]


@eel.expose
def model(*args):
    # 存csv(输入框版本)
    list = ['文化程度', '人均工资', '月末合计', '货币资金', '应收票据', '存货', '固定资产', '总资产', '借款', '应付票据', '应交税金', '实收资本', '负债及权益', '应收帐款',
            '预付账款', '应付账款', '收入', '成本税金', '期间费用', '外收净额', '所得税', '净利润', '毛利率', '净利率', '个人保证', '企业保证', '房产抵押', '知识产权质押',
            '应收账款质押', '营运资金', '净资产', '资产负债率', '流动比率', '速动比率', '长期资产适宜率', '齿轮比率', '年营业收入', '销售利润率', '净资产回报率', '利息保障倍数',
            '净资产增长率', '销售收入增长率', '利润增长率', '利润增长额']

    data = pd.DataFrame([args[0][:44]],columns=list)
    # 交换两列的值
    data['文化程度'], data['月末合计'] = data['月末合计'].copy(), data['文化程度'].copy()

    # data_frames = [pd.DataFrame(arg) for arg in args]
    data.to_csv("data.csv", index=False)
    test_path= "data.csv"


    model_m3e = SentenceTransformer('moka-ai/m3e-large')

    # Our sentences we like to encode
    prospect_code = [prospects]
    strategy_code = [strategy]

    # Sentences are encoded by calling model.encode()
    prospect_embeddings = model_m3e.encode(prospect_code)
    strategy_embeddings = model_m3e.encode(strategy_code)
    prospect_embeddings=pd.DataFrame(prospect_embeddings)
    strategy_embeddings=pd.DataFrame(strategy_embeddings)
    embeddings= pd.concat([prospect_embeddings,strategy_embeddings],axis=1)
    embeddings.columns=[f'PCA{i}' for i in range(1, 2049)]
    # 模型部分
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_id', default='5', type=str)
        parser.add_argument('--task', default='binary', type=str)
        parser.add_argument('--model_name', default=model, type=str)
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--compare', default=False, type=bool)
        all_args = parser.parse_args()
        if method == 'predict':
            output_list = main(all_args,embeddings, method,test_path)
            output_list = output_list[0]
            # 输出list
            if output_list[0] > output_list[1]:
                predict_result = 0
                output = "The prediction is:" + str(
                    predict_result) + ",Demonstration of category prediction probabilities:" + str(
                    output_list) + ",Recommended funding guarantee <300w"
            else:
                predict_result = 1
                output = "The prediction is:" + str(
                    predict_result) + ",Demonstration of category prediction probabilities:" + str(
                    output_list) + ",Recommended funding guarantee >300w"
            return output
        else:
            main(all_args,embeddings, method,test_path)



eel.init(r'.')
eel.start('new.html')