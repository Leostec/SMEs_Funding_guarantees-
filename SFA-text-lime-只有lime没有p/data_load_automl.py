import pandas as pd
import glob
import pickle
import sklearn.preprocessing as preprocessing
from sklearn.experimental import enable_iterative_imputer  # 无此行报错
from sklearn.impute import IterativeImputer
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import string



def datasets_to_X_y(ds_path):
    """
    该函数加载数据集，处理缺失值和分类特征，并将数据集分为特征和目标
    :param ds_path: 数据集路径
    :return: ds_name - 数据集名称
            X - 数据集特征
            y - 数据集目标列
            categorical- 分类特征的索引
            X_oh - 经过独热编码后的数据集特征
    """
    ds_name = '报告数据'  #提取数据集名称
    df = pd.read_csv('D:/研究生/资产评估/提取要素/400大类.csv')  #读取数据集
    # df = df.drop(columns=["报告编号"])
    # df = label_encode_class(df)  #标签换成二分类
    y = df.iloc[:,0]
    X = df.iloc[:,1:]
    column = X.columns.tolist()
    # # 初始化MinMaxScaler
    # scaler0 = MinMaxScaler()
    # # 使用MinMaxScaler对所选列进行归一化
    # X = scaler0.fit_transform(X)
    #标准化
    X=pd.DataFrame(X,columns=column)
    scaler1=StandardScaler()
    X=scaler1.fit_transform(X)
    X= pd.DataFrame(X, columns=column)
    y = y.to_frame()
    # X = X.astype(float)
    # X = impute_missing_values(X)   #多重插补法

    return ds_name, X, y


def impute_missing_values(dataframe):
    """
    该函数使用多重插补法对缺失值进行插补
    :param dataframe: 原始数据集
    :return: temp_dataframe: 插补后的数据集
    """
    temp_dataframe = pd.DataFrame.copy(dataframe, deep=True)  #深度复制（deep=True）传入的原始数据框 dataframe。深度复制确保在对新数据框进行操作时不会影响原始数据框。
    imp = IterativeImputer(max_iter=10, random_state=0)   #创建一个多重插补器对象 imp，使用最大迭代次数为 10 和随机种子为 0。多重插补是一种用于处理缺失值的方法，它通过迭代多次估计缺失值并进行插补
    num_features = temp_dataframe.columns.tolist()  #获取所有数值类型特征的列名
    for col in num_features:
        imp = imp.fit(temp_dataframe[[col]])   #对当前特征列进行插补器的拟合，估计缺失值
        temp_dataframe[col] = imp.transform(temp_dataframe[[col]])   #将估计的缺失值插补到当前特征列中，生成插补后的列
    return temp_dataframe


def label_encode_class(dataframe):
    """
    该函数对目标列进行标签编码
    :param dataframe: 原始数据集
    :return: temp_dataframe: 标签编码后的数据集
    """
    # 根据特定区间替换列的值
    eval_asset_values = dataframe['评估资产'].apply(lambda x: 0 if 0 <= x <= 300 else
    1)
    # 替换原始数据中的评估资产列
    dataframe['评估资产'] = eval_asset_values
    return dataframe


def columns_transform(dataframe):
    """
    该函数对分类列进行独热编码
    :param dataframe: 原始数据集
    :return: temp_dataframe: 经过独热编码后的数据集
    """
    new_cols = []
    binary_data = pd.get_dummies(dataframe)
    for col in binary_data.columns:
        new_cols.append(col.translate(col.maketrans('', '', string.punctuation)))
    binary_data.columns = new_cols
    return binary_data


def read_all_data_files(all_pickle_path, file_pickle_path, files_path):
    """
    该函数获取所有数据集的路径，逐个将它们发送到 datasets_to_X_y 函数进行加载。然后，将它们保存为 pickle 文件
    :param all_pickle_path: 外部 pickle 文件夹的路径
    :param file_pickle_path: pickle 文件保存路径
    :param files_path: 数据集 csv 文件所在的本地路径
    :param ds_id: 当前运行的数据集的 ID
    :param one_hot: 是否使用独热编码来编码数据集
    :return: 与 ds_id 相关的数据集（x、y 和详细信息）
    """
    all_data_idx_name = {}  #all_data_idx_name：一个空字典，用于保存数据集索引信息，包括数据集 ID 和数据集相关的详细信息。
    metadata = {'ds name': [], '# samples': [], '# features': [], '# classes': [], 'class dist': []}  #metadata：一个字典，用于保存数据集的元数据信息，包括数据集名称、样本数量、特征数量、类别数量、类别分布等
    data_file_path = [data_file for data_file in glob.glob(files_path + "/*.csv")]  #通过 glob 模块获取指定路径下所有 .csv 文件的列表。
    if not os.path.exists(file_pickle_path):  #检查 file_pickle_path 路径是否存在，如果不存在则创建。
        os.mkdir(file_pickle_path)
    ds_name, X, y= datasets_to_X_y(data_file_path)   #使用 datasets_to_X_y 函数加载和预处理数据集，得到数据集的名称 ds_name、特征 X、目标列 y、分类特征信息 categorical 和经过独热编码的特征 X_oh（如果适用）
    ds_path_X = file_pickle_path + f'/{ds_name}_X.pkl'
    ds_path_y = file_pickle_path + f'/{ds_name}_y.pkl'     #构建保存数据集特征和目标列的 pickle 文件路径
    with open(ds_path_X, 'wb') as file_X:
        pickle.dump(X, file_X, pickle.HIGHEST_PROTOCOL)
    with open(ds_path_y, 'wb') as file_y:
        pickle.dump(y, file_y, pickle.HIGHEST_PROTOCOL)     #将数据集特征 X 和目标列 y 保存为 pickle 文件

    metadata['ds name'].append(ds_name)
    metadata['# samples'].append(X.shape[0])   #更新元数据信息，添加数据集名称和样本数量
    metadata['# features'].append(X.shape[1])
    #更新元数据信息，添加类别数量、类别分布以及数据集 ID。
    y_array = y.to_numpy().reshape(-1)
    classes = set(y_array)
    metadata['# classes'].append(len(classes))
    class_dist = {i: np.round(list(y_array).count(i)/X.shape[0], 3) for i in classes}
    metadata['class dist'].append(class_dist)
    plot_classes_distribution(ds_name, y_array)  #绘制数据集类别分布图

    #将数据集索引信息添加到 all_data_idx_name 字典中，包括数据集名称、样本数量、特征数量、类别数量和类别分布
    all_data_idx_name = [ds_name, X.shape[0], X.shape[1], len(classes), class_dist]
    #确定要返回的数据集和详细信息

    final_x, final_y, *final_ds_details = X, y, ds_name, X.shape[0], X.shape[1], len(classes), class_dist

    with open(all_pickle_path, 'wb') as all_datasets_names:  #将所有数据集的索引信息保存到外部 pickle 文件中
        pickle.dump(all_data_idx_name, all_datasets_names, pickle.HIGHEST_PROTOCOL)
    #将元数据信息保存到 CSV 文件中
    metadata_df = pd.DataFrame.from_dict(metadata)
    if os.path.exists('metadata.csv'):
        metadata_df.to_csv('metadata.csv', index=False, mode='a', header=False)
    else:
        metadata_df.to_csv('metadata.csv', index=False)
    return final_x, final_y, final_ds_details   #返回指定的数据集特征、目标列和详细信息


def plot_classes_distribution(ds_name, y,):
    """
    该函数绘制数据集类别分布图
    :param ds_name: 数据集名称
    :param y: 类别列
    """
    if not os.path.exists('plots/'):
        os.makedirs('plots/')
    plt.clf()
    plt.hist(y, bins=40, alpha=0.4)
    plt.xticks(list(set(y)))
    plt.xlabel('classes')
    plt.legend(loc='upper left')
    plt.title(f'class distribution- {ds_name}')
    plt.savefig(f'plots/{ds_name}_class_dist.png')

