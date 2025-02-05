from data_load_automl import *
import os
import pickle
import argparse
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from SFALGBMClassifier import SFALGBMClassifier
from SFARandomForestClassifier import SFARandomForestClassifier
from SFAXGBoostClassifier import SFAXGBoostClassifier
# from features_eng import features_tools_extend_X, pca_extend_X
from sklearn.decomposition import PCA

# pickle 是python的一个库，的主要用途是在不同程序或不同时间保存数据状态，以便后续恢复。它在持久化（保存到磁盘）和传递数据时非常有用，
# 特别是在机器学习和数据分析中，可以用来保存训练好的模型、数据集、配置信息等。

# 数据路径和文件名的常量
BINARY_DATA_PATH = 'D:/研究生/资产评估/提取要素'
PICKLE_PATH = './pickle'
BINARY_PICKLE_PATH = PICKLE_PATH + '/binary'
PICKLE_ALL_BINARY_DATASETS_PATH = PICKLE_PATH + '/binary_datasets_idx.pkl'

def main(args):
    # 解析命令行参数
    model_name = args.model_name  #模型名称
    # ds_id = args.dataset_id   #数据集ID
    seed = args.seed   #随机种子
    compare = args.compare   #是否比较

    # 根据任务类型和数据集ID加载数据
    '''load the dataset according to the selected dataset id'''
    pickle_path = BINARY_PICKLE_PATH
    data_path = BINARY_DATA_PATH
    all_datasets_path = PICKLE_ALL_BINARY_DATASETS_PATH
    #读取csv并创建pickle文件，并对数据进行预处理，包括特征工程和模型训练所需的数据划分。
    if not os.path.exists(PICKLE_PATH):
        os.makedirs(PICKLE_PATH, exist_ok=True)
    # read from csv and create pickles
    X, y, ds_details = read_all_data_files(all_datasets_path, pickle_path, data_path)

    print(f'finished datasets load\n')
    if len(ds_details) == 5:
        categories = ds_details[4]
        # ds_details = ds_details[:-1]

    print('seed:', str(seed))

    ds_name = ds_details[0]
    print(f'\nDatatset: {ds_name} \n')
    # 初始化用于 XGBoost、LightGBM 和随机森林的分类器
    models_inits = {'xgb': SFAXGBoostClassifier(ds_details, seed),
                    'lgbm': SFALGBMClassifier(ds_details, seed),
                    'random_forest': SFARandomForestClassifier(ds_details, seed)}

    PEnTex_clf = models_inits[model_name]    #根据给定的模型名称实例化对应的模型，后续以PEnTex_clf为模型变量进行操作
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)  #相同的随机种子使每次运行得到相同的划分，stratify 参数用于分层抽样
                                #，stratify=y，则意味着训练集和测试集中的正负样本比例将与原始数据集中的正负样本比例相同。这有助于防止训练集或测试集中某个类别的数量过于偏差，从而更好地反映现实情况。均有助于在进行对比实验的公平性

    '''optimize hyperparameters'''
    if model_name != 'random_forest':   #对于非随机森林模型，使用超参数优化方法来搜索最佳的超参数组合，并将结果记录下来。
        num_trials, best_trial = PEnTex_clf.run_optimization(X_train, y_train, X_test, y_test)  #run_optimization：该方法在SFAClassifier里面自定义的用于优化超参数
        pre_params = params_dict(best_trial.params)
    else:                       #对于随机森林模型，获取初始超参数设置，并设置数据的类别信息。
        pre_params = PEnTex_clf.get_hyper_params()
        PEnTex_clf.set_categories(categories)
    # 获取最优模型的参数并存入csv
    params_to_set = best_trial.params if model_name != 'random_forest' else pre_params
    PEnTex_clf.set_hyper_params(params_to_set)
    # nepex.log_text('hyper_parameters', str(pre_params))
    hyper_opts_df = pd.DataFrame({
        'Seed': [seed],
        'Dataset': [ds_name],
        'Model': [model_name],
        'HyperOpt params': [pre_params]
    })
    if os.path.exists('hyper_opt.csv'):
        hyper_opts_df.to_csv('hyper_opt.csv', index=False, mode='a', header=False)
    else:
        hyper_opts_df.to_csv('hyper_opt.csv', index=False)

    '''Run SFA'''
    # fit two-step models
    PEnTex_clf.fit(X_train, y_train)  #fit函数是定义在SFAClassifier里的函数，里面包括了SFA的整个流程，得到第一二阶段的值
    # predict
    PEnTex_clf.predict(X_test, y_test)  #predict函数是定义在SFAClassifier里的，作用是通过第一二阶段的模型，计算SFA性能


# 绘制类别分布直方图并保存为图像文件
def plot_classes_distribution(ds_name, y, preds):
    if not os.path.exists('plots/'):
        os.makedirs('plots/')
    plt.clf()
    for i in set(y):
        plt.hist(preds[y[0] == i], bins=40, alpha=0.4, label=str(i))
    plt.legend(loc='upper left')
    plt.title(f'class distribution- {ds_name}')
    plt.savefig(f'plots/{ds_name}_class_dist.png')

# 将超参数字典格式化为易于阅读的形式
def params_dict(best_trial_params):
    params = {}
    for key, value in best_trial_params.items():
        print("    {}: {}".format(key, value))
        params[key] = '{:.4f}'.format(value)
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', default='5', type=str)
    parser.add_argument('--task', default='binary', type=str)
    parser.add_argument('--model_name', default='xgb', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--compare', default=False, type=bool)
    all_args = parser.parse_args()

    main(all_args)



