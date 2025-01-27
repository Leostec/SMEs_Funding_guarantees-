import pdb

import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score as auc
import optuna
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBClassifier, DMatrix
import xgboost as xgb
from lime import submodular_pick
import shap
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

from lightgbm import LGBMClassifier




class SFAClassifier:
    # 初始化与配置参数：在类的构造函数中，初始化了与SFA分类器相关的参数，例如数据集的详细信息（ds_details）、随机种子（seed）、交叉验证的折数（n_folds）以及用于评估模型性能的指标（metric）等。
    def __init__(self,ds_details, seed, n_folds=10, metric='auc'):
        """
        Initialize class parameters
        :param ds_details: the details of the dataset for this run
        :param seed: the seed used for outer and inner splits of the data
        :param n_folds: amount of folds to use in the k fold, 10 is default
        :param metric: the metric used to measure performance, auc is default
        """
        self.ds_name, self.num_samples, self.num_features, self.num_classes, self.class_dist = ds_details
        self.model_name = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.categories = None
        self.seed = seed
        self.params = None
        self.n_folds = n_folds
        self.metric = metric
        self.len_preds = self.num_classes if self.num_classes > 2 else 1

    '''Getter and setters'''
    # 数据处理和操作方法：这个类中包含了许多用于操作和管理数据的方法，例如设置训练数据和测试数据、获取数据的numpy数组表示、设置和获取超参数、获取类别信息等。
    def objective(self, trial):
        return 0

    def set_train_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_test_data(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_y_test_np(self):
        return self.y_test.to_numpy().reshape(-1)

    def get_y_train_np(self):
        return self.y_train.to_numpy()

    def get_X_test_np(self):
        return self.X_test.to_numpy()

    def get_X_train_np(self):
        return self.X_train.to_numpy()

    def set_hyper_params(self, params):
        self.params = params

    def get_hyper_params(self):
        return self.params

    def get_n_classes(self):
        return self.num_classes

    def get_n_features(self):
        return self.num_features

    @staticmethod
    def get_y_np(y):
        return y.to_numpy().reshape(-1)

    @staticmethod
    def get_X_np(X):
        return X.to_numpy()

    def set_categories(self, categories):
        self.categories = categories

    def get_categories(self):
        return self.categories
    # 超参数优化：通过run_optimization方法使用Optuna库进行超参数优化，找到合适的超参数配置，以提升模型性能。
    def run_optimization(self, X_train, y_train, X_test, y_test):
        """
        Optimize hyperparameters using optuna:
        @inproceedings{akiba2019optuna,
        title={Optuna: A next-generation hyperparameter optimization framework},
        author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
        booktitle={Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining},
        pages={2623--2631},
        year={2019}
        }
        :param X_train: train features
        :param y_train: train target column
        :param X_test: test features
        :param y_test: test target column
        :param categories: indices of categorical columns
        :return: num_trials - the number of trials used to find hyperparameters
        :        best_trial - the details of the trial with the best score
        """
        self.set_train_data(X_train, y_train)
        self.set_test_data(X_test, y_test)
        study = optuna.create_study(direction="maximize")  #创建了一个 Optuna Study 对象，用于超参数优化。direction 参数指定了优化的方向，这里是最大化（即寻找最大的性能指标值）
        study.optimize(self.objective, n_trials=15) #optimize 方法来执行优化过程，self.objective 是定义的目标函数，用于评估不同超参数配置的性能。n_trials 参数指定进行的试验次数。
        num_trials = len(study.trials)  #实际进行的试验次数
        best_trial = study.best_trial  #具有最佳分数的试验的详细信息。
        return num_trials, best_trial

#这两个方法分别用于定义实例子样本和特征子样本的下界和上界。它们根据数据集的大小来确定子样本的范围，以用于后续的模型训练
    def get_high_low_subsamples(self):
        """
        Define the lower and upper bounds for the instances subsample in reference to the number of instances
        :return: sub_samples_l - lower bound
                 sub_samples_h - upper bound
        """
        if self.num_samples < 5000:
            sub_samples_l, sub_samples_h = 0.7, 0.95
        elif 5000 <= self.num_samples < 100000:
            sub_samples_l, sub_samples_h = 0.5, 0.85
        else:  # > 100000
            sub_samples_l, sub_samples_h = 0.3, 0.85
        return sub_samples_l, sub_samples_h

    def get_high_low_col_samples(self):
        """
        Define the lower and upper bounds for the features subsample in reference to the number of features
        :return: col_sample_bytree_l - lower bound
                 col_sample_bytree_h - upper bound
        """
        if self.num_features < 50:
            col_sample_bytree_l, col_sample_bytree_h = 0.3, 1
        elif 50 <= self.num_features < 500:
            col_sample_bytree_l, col_sample_bytree_h = 0.6, 1
        else:
            col_sample_bytree_l, col_sample_bytree_h = 0.15, 0.8
        return col_sample_bytree_l, col_sample_bytree_h

    #用于训练整个SFA模型的两个阶段，并保存模型。
    def fit(self, X_train, y_train):
        """
        Train the SFA models in two stages and save them.
        :param X_train: train data
        :param y_train: train target
        """
        # Train first-stage model
        X_train_data = X_train.iloc[:, :44]
        X_train_pca = X_train.iloc[:, 44:]
        # 该方法返回了第一阶段模型在验证集上的预测值 val_preds、对应的lime值 val_shap_values
        val_lime_values, X_train_lime = self.train_first_stage(X_train_data, y_train)
        # use the OOP predictions and Shapley values to create 3 variations of augmented features
        # 通过使用预测值和SHAP值来创建三种不同的增强特征：p-shap、shap 和 p
        train_df_lime = pd.DataFrame(val_lime_values, columns=[f'shap_{col}' for col in X_train_data.columns],
                                     index=X_train_data.index)  # 将 val_shap_values 转化为 train_df_shap 数据框，列名为 shap_{col}，行索引与训练数据一致
        # column = train_df_lime.columns.tolist()
        # scaler = StandardScaler()
        # train_df_lime = scaler.fit_transform(train_df_lime)
        # train_df_lime = pd.DataFrame(train_df_lime, columns=column)

        # Train 3 second-stage models 在这三种增强特征上训练第二阶段的模型
        key_list = self.train_second_stage(X_train_data, y_train, 'base', 40)
        key_list_lime = self.train_second_stage(train_df_lime, y_train, 'lime', 8)

        # 只保留lime中贡献排名前i的特征
        train_df_lime = train_df_lime.drop(columns=train_df_lime.columns.difference(key_list_lime))
        # 保留base样本中贡献排名前i的特征
        X_train_data = X_train_data.drop(columns=X_train_data.columns.difference(key_list))

        # 再次通过使用 join 方法，将原始的训练数据 X_train 与 train_df_lime、train_df_preds 进行合并，得到三种增强特征
        X_train_ex_lime = X_train_data.join(train_df_lime)  # lime
        X_train_ex_lime_text = X_train_ex_lime.join(X_train_pca)
        X_train_ex_text = X_train_data.join(X_train_pca)
        # Train 3 second-stage models 在这三种增强特征上训练第二阶段的模型
        self.train_second_stage(X_train_ex_lime_text, y_train, 'base-lime-text', 0)
        self.train_second_stage(X_train_ex_lime, y_train, 'lime', 0)
        self.train_second_stage(X_train_pca, y_train, 'text', 0)
        self.train_second_stage(X_train_ex_text, y_train, 'base-text', 0)
        return X_train_lime, key_list_lime, key_list

    # 第一阶段模型训练：train_first_stage方法用于训练第一阶段的基础模型，并计算出-of-fold（OOF）预测和对应的SHAP值，用于生成增强的特征。
    def train_first_stage(self, X_train_val, y_train_val):
        """
        Train the first-stage models (base model) using k-fold cross validation. Save the models in .model form.
        Calculate the OOF predictions and their corresponding SHAP values using TreeExplainer for the second stage.
        @article{lundberg2020local,
        title={From local explanations to global understanding with explainable AI for trees},
        author={Lundberg, Scott M and Erion, Gabriel and Chen, Hugh and DeGrave, Alex and Prutkin, Jordan M and Nair,
         Bala and Katz, Ronit and Himmelfarb, Jonathan and Bansal, Nisha and Lee, Su-In},
        journal={Nature machine intelligence},
        volume={2},
        number={1},
        pages={56--67},
        year={2020},
        publisher={Nature Publishing Group}
        }
        :param X_train_val: train + validation data
        :param y_train_val: train + validation target
        :return:
        """
        # 使用 Stratified K-Fold 交叉验证方法划分为多个折
        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        y_train_np = self.get_y_np(y_train_val)  #将y转换为numpy数组

        # 初始化存储在第一阶段训练过程中的结果的数组（创造对应形状的全0数组:np.zeros()）
        val_preds = np.zeros((X_train_val.shape[0], self.len_preds)) #被用于存储基于第一阶段模型的模型预测结果
        val_all_predicitions = np.zeros(X_train_val.shape[0]) #用于存储每个验证集样本的预测类别
        val_all_probas = np.zeros((X_train_val.shape[0], self.num_classes)) #用于存储每个验证集样本的预测概率值
        val_lime_vals = np.zeros((X_train_val.shape[0], X_train_val.shape[1]))  #用于存储每个验证集样本的 lime 值
        # val_feature_importance=np.zeros()
        N=0
        explainer = LimeTabularExplainer(X_train_val.values, training_labels=y_train_val,
                                         mode="classification")  # 创建一个lime 解释器。
        #循环对kf每一折都进行操作
        for i, (tr_ind, val_ind) in enumerate(kf.split(X_train_val, y_train_val)):
            # 分成训练集和验证集
            X_train, y_train, X_val, y_val = X_train_val.iloc[tr_ind], y_train_val.iloc[tr_ind], \
                                             X_train_val.iloc[val_ind], y_train_val.iloc[val_ind]
            #初始化并训练基于树的分类器（第一阶段模型）
            clf = self.train(X_train, y_train)  #train函数定义在SFARandomForestClassifier和另外两个模型文件中

            # save the trained model
            if not os.path.exists('models'):  #检查当前路径下是否有models文件夹，若没有则用makedirs构建一个。
                os.makedirs('models', exist_ok=True)  #exist_ok=True表示若文件夹存在则不会发生异常
            if not os.path.exists(f'models/{self.ds_name}'):  #检查当前目录下是否存在名为 self.ds_name（数据集名称）的文件夹。
                os.makedirs(f'models/{self.ds_name}', exist_ok=True)
            if not os.path.exists(f'models/{self.ds_name}/{self.model_name}'):  #检查当前目录下是否存在名为 self.model_name（模型名称）的文件夹。
                os.makedirs(f'models/{self.ds_name}/{self.model_name}', exist_ok=True)
            self.save_model(clf, f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}')  #调用 self.save_model 方法来保存训练好的模型。 该方法定义在SFARandomForestClassifier和另外两个模型文件中
            # predict on validation
            probabilities = self.predict_proba_train(clf,(X_val,y_val))  #使用模型 clf 对验证集 X_val 进行预测，得到每个类别的概率值。predict_proba 定义在SFARFC里面，是用来计算预测概率值

            prediction = probabilities.argmax(axis=1)  #对于每个样本，找到预测概率最高的类别，得到一个数组 prediction，表示模型预测的类别。
            val_all_probas[val_ind, :] = probabilities   #将所有类别的预测概率值都存储在 val_all_probas 数组中。
            val_all_predicitions[val_ind] = prediction   #将每个样本的预测值存储在 val_all_predicitions 数组中。
            # calculate lime values for the validation
            # X_val=np.array(X_val)
            if self.model_name in ['xgb', 'random_forest']:
                def predict_fn(x):
                    x = xgb.DMatrix(x)
                    preds = clf.predict(x).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                path = f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}'
                clf=self.load_model(path)

                sp_obj = submodular_pick.SubmodularPick(explainer, X_val.values, predict_fn,method='full', num_features=len(X_val.columns))
                for i, exp in enumerate(sp_obj.explanations):
                    # l = exp.as_list()
                    i = i + N
                    for j in range(X_train_val.shape[1]):
                        # pdb.set_trace()
                        for x in exp.local_exp.keys():
                            # pdb.set_trace()
                            idx = exp.local_exp[x][j][0]
                            # idx= sp_obj.explanations[3].local_exp[x][j][0]
                            # if exp.local_exp[x][j][1] >0:
                            val_lime_vals[i, idx] = exp.local_exp[x][j][1]  # 存储每个样本的特征贡献值
                N = N + len(val_ind)
            else:
                def predict_fn(x):
                    preds = clf.predict(x).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                path = f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}'
                clf=self.load_model(path)

                # lime值提取法三---splime
                sp_obj = submodular_pick.SubmodularPick(explainer, X_val.values, predict_fn,method='full', num_features=len(X_val.columns))
                for i, exp in enumerate(sp_obj.explanations):
                    # l = exp.as_list()
                    i = i + N
                    for j in range(X_train_val.shape[1]):
                        # pdb.set_trace()
                        for x in exp.local_exp.keys():
                            idx = exp.local_exp[x][j][0]
                            val_lime_vals[i, idx] = exp.local_exp[x][j][1]  # 存储每个样本的特征贡献值
                N=N + len(val_ind)
                # pdb.set_trace()
        # 计算评估指标
        if self.metric == 'auc': #这行代码首先检查模型的评估指标是否为'AUC'
            val_score = float('{:.4f}'.format(auc(y_train_np, val_all_probas, multi_class='ovo') if  #val_score，用于存储计算得到的验证分数
                                              self.get_n_classes() > 2 else auc(y_train_np, val_all_probas[:, 1])))
        elif self.metric == 'logloss':
            val_score = float('{:.4f}'.format(log_loss(y_train_np, val_all_probas)))
        print('base val score', str(val_score))

        return val_lime_vals, X_train_val


    # 第二阶段模型训练：train_second_stage方法用于在增强的特征上训练第二阶段的模型，其中包括三个变种：P增强、SHAP增强和P+SHAP增强。
    def train_second_stage(self, X_train_ext, y_train, config,i):
        """
        Train the second-stage model on the augmented features
        :param X_train_ext: train augmented features
        :param y_train: train target
        :param config: augmented data variation (P augmented, SHAP augmented or P+SHAP augmented)
        :return:
        """
        clf = self.train(X_train_ext, y_train)
        self.save_model(clf, f'models/{self.ds_name}/{self.model_name}/meta_{config}_seed_{self.seed}')  #调用save_model方法将训练后的模型clf保存到指定的文件路径，其中{self.ds_name}和{self.model_name}是数据集名称和模型名称，{config}是配置信息，{self.seed}是随机种子。
        preds = self.predict_proba_train(clf, (X_train_ext, y_train))  #使用训练好的模型对扩展的训练集X_train_ext进行预测，得到预测的概率值。预测结果存储在变量preds中。
        y_train_np = self.get_y_np(y_train)  #将训练集的目标标签y_train转换为NumPy数组
        #同样计算性能指标
        if self.metric == 'auc':
            train_score = float('{:.4f}'.format(auc(y_train_np, preds, multi_class='ovo') if self.get_n_classes() > 2
                                                else auc(y_train_np, preds[:, 1])))
        elif self.metric == 'logloss':
            train_score = float('{:.4f}'.format(log_loss(y_train_np, preds)))
        print((f'train meta score- {config}', str(train_score)))
        if self.model_name in ['xgb', 'random_forest']:
            imp = clf.get_fscore()
            # 根据字典的值进行降序排序
            sorted_dict = dict(sorted(imp.items(), key=lambda item: item[1], reverse=True))
            print(sorted_dict)
            # if 'shap' in imp:
            if config == 'lime':
                imp_10 = {key:imp[key] for key in imp if 'shap' in key}
                imp = sorted(imp_10.items(),key = lambda x:x[1],reverse=True)[:i]
                key_list = [item[0] for item in imp]
                return key_list

            else:
                imp_10 = {key: imp[key] for key in imp}
                imp = sorted(imp_10.items(), key=lambda x: x[1], reverse=True)[:i]
                key_list = [item[0] for item in imp]
                return key_list
        else:
            imp=clf.feature_importance()
            feature_names = X_train_ext.columns
            # 转化为字典
            imp = {feature: importance for feature, importance in
                                       zip(feature_names, imp)}
            # if 'shap' in imp:
            if config == 'lime':
                imp_10 = {key:imp[key] for key in imp if 'shap' in key}
                imp = sorted(imp_10.items(),key = lambda x:x[1],reverse=True)[:i]
                key_list = [item[0] for item in imp]
                return key_list
            else:
                imp_10 = {key: imp[key] for key in imp}
                imp = sorted(imp_10.items(), key=lambda x: x[1], reverse=True)[:i]
                key_list = [item[0] for item in imp]
                return key_list



    # 预测和性能评估：predict方法用于预测测试集的得分，使用第一阶段和第二阶段的模型，计算SFA分类器的性能评分。
    def predict(self, X_train, X_test,embedding, key_list_lime, key_list):
        """
        Predict the score for the test set using the trained first-stage and second-stage models
        :param X_test: test features
        :param y_test: test target
        :return: SFA score
        """
        X_test_data = X_test
        X_test_pca = embedding
        # predict using the first-stage model
        test_lime, test_all_probas = self.predict_first_stage(X_train,X_test_data)   #使用第一阶段模型对测试集进行预测，得到原始预测结果 test_preds、Shapley 值 test_shap 和所有类别的概率预测 test_all_probas

        # use the OOP predictions and Shapley values to create 3 variations of augmented features
        test_df_lime = pd.DataFrame(data=test_lime, columns=[f'shap_{col}' for col in X_test_data.columns],
                                    index=X_test_data.index)
        # column = test_df_lime.columns.tolist()
        # scaler = StandardScaler()
        # test_df_lime = scaler.fit_transform(test_df_lime)
        # test_df_lime = pd.DataFrame(test_df_lime, columns=column)
        # 只保留lime中贡献排名前十的特征
        test_df_lime = test_df_lime.drop(columns= test_df_lime.columns.difference(key_list_lime))
        # 保留base样本中贡献排名前i的特征
        X_test_data = X_test_data.drop(columns=X_test_data.columns.difference(key_list))
        X_test_ex_lime = X_test_data.join(test_df_lime)  # lime
        X_test_ex_lime_text = X_test_ex_lime.join(X_test_pca)
        X_test_ex_text = X_test_data.join(X_test_pca)
        # predict using the second_stage model
        preds_lime = self.predict_second_stage(X_test_ex_lime_text,'base-lime-text')   #使用第二阶段模型对使用 Shapley 值特征的测试集进行预测，得到预测结果 preds_shap
        # self.predict_second_stage(test_df_lime, y_test, 'onlylime')  #返回只有shap值的
        total_score_mean = self.calc_average_test_score(test_all_probas, #使用预测概率、三种预测结果和真实标签计算测试集的平均分数
                                                              [preds_lime])
        self.predict_second_stage(X_test_pca, 'text')
        self.predict_second_stage(X_test_ex_lime, 'lime')
        self.predict_second_stage(X_test_ex_text,  'base-text')
        print(f'SFA test score', str(total_score_mean))
        return total_score_mean

    def predict_first_stage(self,X_train, X_test):
        """
        Predict score for the test set using the k first-stage models. For each model - load it and calculate prediction and SHAP values.
        Also calculate and print metric value.
        :param X_test: test features
        :param y_test: test target
        :return: avg_test_preds - the average (probability) prediction for each instance
                 avg_test_all_probas - the average (probability) prediction for each instance of all class if multi class, of the positive class if binary
                 avg_test_shap -  average SHAP values for each instance
        """
        test_all_probas = np.zeros((self.n_folds, X_test.shape[0], self.num_classes))  #创建用来存储每折中模型的类别概率预测
        test_lime_vals = np.zeros((self.n_folds, X_test.shape[0], X_test.shape[1]))  #创建用来存储每折中模型的 SHAP 值

        for i in range(self.n_folds):  #循环遍历每折
            path = f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}'
            clf = self.load_model(path)             #加载训练好的模型
            # prediction
            model_preds = self.predict_proba(clf, X_test)   #进行预测并得到模型的预测结果 model_preds 和类别预测 model_prediction

            model_prediction = model_preds.argmax(axis=1)   #找到每个样本预测值最大的一列（类别）

            explainer = LimeTabularExplainer(X_train.values,mode="classification")  # 创建一个lime 解释器。

            if self.model_name in ['xgb', 'random_forest']:
                def predict_fn(x):
                    x = xgb.DMatrix(x)
                    preds = clf.predict(x).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                # clf = xgb.Booster()
                path = f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}'
                clf=self.load_model(path)
                sp_obj = submodular_pick.SubmodularPick(explainer, X_test.values, predict_fn,method='full', num_features=len(X_test.columns))
                for n, exp in enumerate(sp_obj.explanations):
                    # l = exp.as_list()
                    for j in range(X_test.shape[1]):
                        # pdb.set_trace()
                        for x in exp.local_exp.keys():
                            idx = exp.local_exp[x][j][0]
                            # if exp.local_exp[x][j][1] > 0:
                            test_lime_vals[i,n, idx] = exp.local_exp[x][j][1]  # 存储每个样本的特征贡献值
                # explanation = explainer.explain_instance(X_test.iloc[i].values, clf, num_features=len(y_test))
            else:
                def predict_fn(x):
                    preds = clf.predict(x).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                path = f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}'
                clf=self.load_model(path)
                # lime值提取法三---splime
                sp_obj = submodular_pick.SubmodularPick(explainer, X_test.values, predict_fn,method='full', num_features=len(X_test.columns))
                for n, exp in enumerate(sp_obj.explanations):
                    for j in range(X_test.shape[1]):
                        for x in exp.local_exp.keys():
                            idx = exp.local_exp[x][j][0]
                            test_lime_vals[i, n, idx] = exp.local_exp[x][j][1]  # 存储每个样本的特征贡献值
            test_all_probas[i, :, :] = model_preds
        #计算所有折内的预测结果、类别概率和SHAP值的平均值
        avg_test_all_probas = np.mean(test_all_probas, axis=0)
        avg_test_shap = np.mean(test_lime_vals, axis=0)

        return avg_test_shap, avg_test_all_probas

    def predict_second_stage(self, X_test, config):
        """
         Predict score for the test set using the second-stage model according to config (either P augmented model, SHAP augmented model or P+SHAP augmented model).
         Load the model and calculate prediction and metric value.
        :param X_test: test features
        :param y_test: test target
        :param config: the name of the augmented model to load
        :return: probability predictions
        """
        total_dict={}
        path = f'models/{self.ds_name}/{self.model_name}/meta_{config}_seed_{self.seed}'
        clf = self.load_model(path)                          #加载模型
        preds = self.predict_proba(clf, X_test)    #对数据进行预测                    #将y_test转换为np
        ap= clf.predict(DMatrix(X_test))
        print(ap)
        if config == "lime":
            # 创建一个SHAP解释器
            feature_names_english = ['Educational level', 'Per capita wages', 'Month-end total',
                                     'Cash and cash equivalents', 'Notes receivable', 'Inventories',
                                     'Fixed assets', 'Total assets', 'Borrowing', 'Notes payable', 'Tax payable',
                                     'Paid-up capital', 'Liabilities and equity',
                                     'Accounts receivable', 'Prepayment', 'Accounts payable', 'Income', 'Costs and taxes',
                                     'Period expense', 'Net external receipts',
                                     'Income tax', 'Net profit', 'Gross margin', 'Net profit ratio', 'Personal guarantee',
                                     'Corporate guarantee', 'Real estate mortgage',
                                     'Intellectual Property Pledge', 'Pledge of accounts receivable', 'Working assets',
                                     'Net assets', 'Debt asset ratio', 'Current ratio',
                                     'Quick ratio', 'Long-term assets suitability ratio', 'Gear ratio',
                                     'Annual operating income', 'Profit-volume margin', 'Return on Net Assets',
                                     'Interest coverage multiple', 'Net asset growth rate', 'Growth rate of sales revenue',
                                     'Growth rate of profitability', 'Amount of profit growth']

            feature_names_chinese = ['文化程度', '人均工资', '月末合计', '货币资金', '应收票据', '存货', '固定资产', '总资产', '借款', '应付票据', '应交税金',
                                     '实收资本', '负债及权益', '应收帐款',
                                     '预付账款', '应付账款', '收入', '成本税金', '期间费用', '外收净额', '所得税', '净利润', '毛利率', '净利率', '个人保证',
                                     '企业保证', '房产抵押', '知识产权质押',
                                     '应收账款质押', '营运资金', '净资产', '资产负债率', '流动比率', '速动比率', '长期资产适宜率', '齿轮比率', '年营业收入', '销售利润率',
                                     '净资产回报率', '利息保障倍数',
                                     '净资产增长率', '销售收入增长率', '利润增长率', '利润增长额']

            feature_names_english = ['Educational level', 'Per capita wages', 'Month-end total',
                                     'Cash and cash equivalents', 'Notes receivable', 'Inventories',
                                     'Fixed assets', 'Total assets', 'Borrowing', 'Notes payable', 'Tax payable',
                                     'Paid-up capital', 'Liabilities and equity',
                                     'Accounts receivable', 'Prepayment', 'Accounts payable', 'Income', 'Costs and taxes',
                                     'Period expense', 'Net external receipts',
                                     'Income tax', 'Net profit', 'Gross margin', 'Net profit ratio', 'Personal guarantee',
                                     'Corporate guarantee', 'Real estate mortgage',
                                     'Intellectual Property Pledge', 'Pledge of accounts receivable', 'Working assets',
                                     'Net assets', 'Debt asset ratio', 'Current ratio',
                                     'Quick ratio', 'Long-term assets suitability ratio', 'Gear ratio',
                                     'Annual operating income', 'Profit-volume margin', 'Return on Net Assets',
                                     'Interest coverage multiple', 'Net asset growth rate', 'Growth rate of sales revenue',
                                     'Growth rate of profitability', 'Amount of profit growth']

            feature_names_chinese = ['文化程度', '人均工资', '月末合计', '货币资金', '应收票据', '存货', '固定资产', '总资产', '借款', '应付票据', '应交税金',
                                     '实收资本', '负债及权益', '应收帐款',
                                     '预付账款', '应付账款', '收入', '成本税金', '期间费用', '外收净额', '所得税', '净利润', '毛利率', '净利率', '个人保证',
                                     '企业保证', '房产抵押', '知识产权质押',
                                     '应收账款质押', '营运资金', '净资产', '资产负债率', '流动比率', '速动比率', '长期资产适宜率', '齿轮比率', '年营业收入', '销售利润率',
                                     '净资产回报率', '利息保障倍数',
                                     '净资产增长率', '销售收入增长率', '利润增长率', '利润增长额']

            # lime_eng = ["lime_" + english_name for english_name in feature_names_english]
            # lime_chn = ["shap_" + chinese_name for chinese_name in feature_names_chinese]
            # name_mapping = {chinese_name: english_name for chinese_name, english_name in
            #                 zip(feature_names_chinese, feature_names_english)}
            #
            # # 将未加前缀的列名也加入字典
            # name_mapping.update({chinese_name: english_name for chinese_name, english_name in zip(lime_chn, lime_eng)})
            #
            feature_names = X_test.columns

            # 使用之前创建的字典进行映射
            # feature_names = [name_mapping.get(name, name) for name in feature_names]
            explainer = shap.TreeExplainer(clf)
            # 计算SHAP值
            shap_values = explainer.shap_values(X_test)

            force_plot=shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0], feature_names=feature_names)
            shap.save_html('force_plot.html', force_plot)
        return preds

    def calc_average_test_score(self, test_preds_base, list_preds):
        avg_preds = np.mean([test_preds_base] + list_preds, axis=0)

        return avg_preds

    '''compare to featuretools and pca augment'''
    # todo: remove for production - experiments only
    def train_other(self, X_train, y_train, other_name):
        """
        Train model using k fold cross validation on augmented features created using a different augmentation method
        :param X_train: augmented train features
        :param y_train: train target
        :param other_name: augmentation method's name
        """
        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        for i, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            train_x, val_x, train_y, val_y = X_train.iloc[tr_idx], X_train.iloc[val_idx], \
                                             y_train.iloc[tr_idx], y_train.iloc[val_idx]
            clf = self.train(train_x, train_y)
            if not os.path.exists(f'models_{other_name}'):
                os.makedirs(f'models_{other_name}', exist_ok=True)
            if not os.path.exists(f'models_{other_name}/{self.ds_name}'):
                os.makedirs(f'models_{other_name}/{self.ds_name}', exist_ok=True)
            if not os.path.exists(f'models_{other_name}/{self.ds_name}/{self.model_name}'):
                os.makedirs(f'models_{other_name}/{self.ds_name}/{self.model_name}', exist_ok=True)
            self.save_model(clf, f'models_{other_name}/{self.ds_name}/{self.model_name}/fold_{i}_seed_{self.seed}')

    # todo: remove for production - experiments only
    def predict_other(self, X_test, y_test, other_name):
        """
        Predict using the k models trained on augmented features created using a different augmentation method.
        Also calculate and print metric value.
        :param X_test: augmented test features
        :param y_test: test target
        :param other_name:
        :return: other_name: augmentation method's name
        """
        y_test_np = self.get_y_np(y_test)
        test_all_auc = []
        for i in range(self.n_folds):
            path = f'models_{other_name}/{self.ds_name}/{self.model_name}/fold_{i}_seed_{self.seed}'
            clf = self.load_model(path)
            model_preds = self.predict_proba(clf, (X_test, y_test))

            if self.metric == 'auc':
                score = auc(y_test_np, model_preds, multi_class='ovo') if self.get_n_classes() > 2 else \
                    auc(y_test_np, model_preds[:, 1])
            elif self.metric == 'logloss':
                score = log_loss(y_test_np, model_preds)
            test_all_auc.append(score)

            test_score = float('{:.4f}'.format(np.mean(test_all_auc, axis=0)))
            print(f'{other_name} test score', str(test_score))



