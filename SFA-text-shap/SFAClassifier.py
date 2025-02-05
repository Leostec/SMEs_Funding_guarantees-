import numpy as np
import os
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score as auc
import optuna
import pandas as pd

import SFALGBMClassifier

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
        # 该方法返回了第一阶段模型在验证集上的预测值 val_preds、对应的SHAP值 val_shap_values
        val_preds, val_shap_values = self.train_first_stage(X_train, y_train)

        # use the OOP predictions and Shapley values to create 3 variations of augmented features
        # 通过使用预测值和SHAP值来创建三种不同的增强特征：p-shap、shap 和 p
        train_df_shap = pd.DataFrame(val_shap_values, columns=[f'shap_{col}' for col in X_train.columns],
                                     index=X_train.index)      #将 val_shap_values 转化为 train_df_shap 数据框，列名为 shap_{col}，行索引与训练数据一致
        train_df_preds = pd.DataFrame(val_preds, columns=[f'preds_{i}' for i in range(self.len_preds)], index=X_train.index)       #将 val_preds 转化为 train_df_preds 数据框，列名为 preds_{i}，行索引与训练数据一致。然后，通过使用 join 方法

        #通过使用 join 方法，将原始的训练数据 X_train 与 train_df_shap、train_df_preds 进行合并，得到三种增强特征
        X_train_ex_p_shap = X_train.join(train_df_shap).join(train_df_preds)  # p-shap
        X_train_ex_shap = X_train.join(train_df_shap)  # shap
        X_train_ex_p = X_train.join(train_df_preds)  # p

        # Train 3 second-stage models 在这三种增强特征上训练第二阶段的模型
        self.train_second_stage(X_train_ex_p, y_train, 'p')
        self.train_second_stage(X_train_ex_shap, y_train, 'shap')
        self.train_second_stage(X_train_ex_p_shap, y_train, 'p_shap')
        self.train_second_stage(train_df_shap,y_train,'onlyshap')
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
        val_shap_vals = np.zeros(X_train_val.shape)  #用于存储每个验证集样本的 SHAP 值

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
            probabilities = self.predict_proba(clf, (X_val, y_val))  #使用模型 clf 对验证集 X_val 进行预测，得到每个类别的概率值。predict_proba 定义在SFARFC里面，是用来计算预测概率值
            prediction = probabilities.argmax(axis=1)  #对于每个样本，找到预测概率最高的类别，得到一个数组 prediction，表示模型预测的类别。
            val_preds[val_ind, :] = probabilities if self.len_preds > 1 else \
                    probabilities[:, 1].reshape(probabilities.shape[0], 1) # 将预测的概率值存储在 val_preds 数组中，如果有多个类别，则存储所有概率值，否则只存储概率最高的类别的概率值。
            val_all_probas[val_ind, :] = probabilities   #将所有类别的预测概率值都存储在 val_all_probas 数组中。
            val_all_predicitions[val_ind] = prediction   #将每个样本的预测值存储在 val_all_predicitions 数组中。
            # calculate SHAP values for the validation
            clf_ex = shap.TreeExplainer(clf)   #创建一个用于解释树模型的 SHAP 解释器。
            if self.model_name in ['xgb', 'random_forest'] and self.categories is not None:   #检查当前模型是否为 XGBoost 或随机森林，并且是否存在分类信息。
                dvalid = self.get_DMatrix(X_val, y_val)   #如果模型是 XGBoost，将验证集转换为 DMatrix 格式，以便进行 SHAP 值计算
                shap_values = clf_ex.shap_values(dvalid, check_additivity=False)   #计算验证集的 SHAP 值，得到一个二维数组，其中每行对应一个样本，每列对应一个特征，表示每个特征对预测结果的贡献。
            else:
                shap_values = clf_ex.shap_values(X_val, check_additivity=False)  #对于多分类问题，根据预测的类别，从 SHAP 值中选择对应的类别的解释性值。
            val_shap_vals[val_ind] = [shap_values[height_idx][j] for j, height_idx in enumerate(prediction)] if (
                        self.get_n_classes() > 2 or self.model_name == 'lgbm') \
                else shap_values

        # 计算评估指标
        if self.metric == 'auc': #这行代码首先检查模型的评估指标是否为'AUC'
            val_score = float('{:.4f}'.format(auc(y_train_np, val_all_probas, multi_class='ovo') if  #val_score，用于存储计算得到的验证分数
                                              self.get_n_classes() > 2 else auc(y_train_np, val_all_probas[:, 1])))
        elif self.metric == 'logloss':
            val_score = float('{:.4f}'.format(log_loss(y_train_np, val_all_probas)))
        print('base val score', str(val_score))

        return val_preds, val_shap_vals
    # 第二阶段模型训练：train_second_stage方法用于在增强的特征上训练第二阶段的模型，其中包括三个变种：P增强、SHAP增强和P+SHAP增强。
    def train_second_stage(self, X_train_ext, y_train, config):
        """
        Train the second-stage model on the augmented features
        :param X_train_ext: train augmented features
        :param y_train: train target
        :param config: augmented data variation (P augmented, SHAP augmented or P+SHAP augmented)
        :return:
        """
        clf = self.train(X_train_ext, y_train)
        self.save_model(clf, f'models/{self.ds_name}/{self.model_name}/meta_{config}_seed_{self.seed}')  #调用save_model方法将训练后的模型clf保存到指定的文件路径，其中{self.ds_name}和{self.model_name}是数据集名称和模型名称，{config}是配置信息，{self.seed}是随机种子。
        preds = self.predict_proba(clf, (X_train_ext, y_train))  #使用训练好的模型对扩展的训练集X_train_ext进行预测，得到预测的概率值。预测结果存储在变量preds中。
        y_train_np = self.get_y_np(y_train)  #将训练集的目标标签y_train转换为NumPy数组
        #同样计算性能指标
        if self.metric == 'auc':
            train_score = float('{:.4f}'.format(auc(y_train_np, preds, multi_class='ovo') if self.get_n_classes() > 2
                                                else auc(y_train_np, preds[:, 1])))
        elif self.metric == 'logloss':
            train_score = float('{:.4f}'.format(log_loss(y_train_np, preds)))
        print((f'train meta score- {config}', str(train_score)))
    # 预测和性能评估：predict方法用于预测测试集的得分，使用第一阶段和第二阶段的模型，计算SFA分类器的性能评分。
    def predict(self, X_test, y_test):
        """
        Predict the score for the test set using the trained first-stage and second-stage models
        :param X_test: test features
        :param y_test: test target
        :return: SFA score
        """
        # predict using the first-stage model
        test_preds, test_shap, test_all_probas = self.predict_first_stage(X_test, y_test)   #使用第一阶段模型对测试集进行预测，得到原始预测结果 test_preds、Shapley 值 test_shap 和所有类别的概率预测 test_all_probas

        # use the OOP predictions and Shapley values to create 3 variations of augmented features
        test_df_preds = pd.DataFrame(test_preds, columns=[f'preds_{i}' for i in range(self.len_preds)],  #将原始预测结果和 Shapley 值转换为 DataFrame 格式
                                     index=X_test.index)
        test_df_shap = pd.DataFrame(test_shap, columns=[f'shap_{col}' for col in X_test.columns],
                                    index=X_test.index)

        X_test_ex_p_shap = X_test.join(test_df_shap).join(test_df_preds)  # p-shap
        X_test_ex_shap = X_test.join(test_df_shap)  # shap
        X_test_ex_p = X_test.join(test_df_preds)  # p

        # predict using the second_stage model
        preds_p = self.predict_second_stage(X_test_ex_p, y_test, 'p')   #使用第二阶段模型对使用原始预测结果特征的测试集进行预测，得到预测结果 preds_p
        preds_shap = self.predict_second_stage(X_test_ex_shap, y_test, 'shap')   #使用第二阶段模型对使用 Shapley 值特征的测试集进行预测，得到预测结果 preds_shap
        preds_p_shap = self.predict_second_stage(X_test_ex_p_shap, y_test, 'p_shap')  #使用第二阶段模型对使用同时包含原始预测结果和 Shapley 值特征的测试集进行预测，得到预测结果 preds_p_shap
        self.predict_second_stage(test_df_shap, y_test, 'onlyshap')  #返回只有shap值的
        total_score_mean = self.calc_average_test_score(test_all_probas, [preds_p,  #使用预测概率、三种预测结果和真实标签计算测试集的平均分数
                                                              preds_shap, preds_p_shap], y_test)
        print(f'SFA test score', str(total_score_mean))
        return total_score_mean

    def predict_first_stage(self, X_test, y_test):
        """
        Predict score for the test set using the k first-stage models. For each model - load it and calculate prediction and SHAP values.
        Also calculate and print metric value.
        :param X_test: test features
        :param y_test: test target
        :return: avg_test_preds - the average (probability) prediction for each instance
                 avg_test_all_probas - the average (probability) prediction for each instance of all class if multi class, of the positive class if binary
                 avg_test_shap -  average SHAP values for each instance
        """
        test_preds = np.zeros((self.n_folds, X_test.shape[0], self.len_preds))  #创建用来存储每折中模型的预测结果
        test_all_probas = np.zeros((self.n_folds, X_test.shape[0], self.num_classes))  #创建用来存储每折中模型的类别概率预测
        test_shap_vals = np.zeros((self.n_folds, X_test.shape[0], X_test.shape[1]))  #创建用来存储每折中模型的 SHAP 值
        all_test_score = np.zeros(self.n_folds)  #创建用来存储每折的测试分数
        y_test_np = self.get_y_np(y_test)   #将测试集标签转换为 NumPy 数组

        for i in range(self.n_folds):  #循环遍历每折
            path = f'models/{self.ds_name}/{self.model_name}/base_fold_{i}_seed_{self.seed}'
            clf = self.load_model(path)             #加载训练好的模型
            # prediction
            model_preds = self.predict_proba(clf, (X_test, y_test))   #进行预测并得到模型的预测结果 model_preds 和类别预测 model_prediction
            model_prediction = model_preds.argmax(axis=1)  #找到每个样本预测值最大的一列（类别）
            # SHAP values  计算SHAP值
            clf_ex = shap.TreeExplainer(clf)  #创建一个 SHAP TreeExplainer 对象，用于解释由 clf 模型进行预测的数据
            if self.model_name in ['xgb', 'random_forest'] and self.categories is not None:   #检查当前模型的名称是否在 ['xgb', 'random_forest'] 列表中，并且是否存在分类信息
                dtest = self.get_DMatrix(X_test, y_test)    #调用 self.get_DMatrix 方法（定义在XGBoostClassifier和其他两个类里面）将测试集数据 X_test 和 y_test 转换为 DMatrix 对象（是 XGBoost 用于存储和处理训练和验证数据的）。
                shap_values = clf_ex.shap_values(dtest, check_additivity=False)  #使用 TreeExplainer 对象计算测试集数据的 SHAP 值。check_additivity 参数设置为 False，表示不进行加性检验（additivity check）
            else:
                shap_values = clf_ex.shap_values(X_test, check_additivity=False) #直接使用 TreeExplainer 对象计算测试集数据 X_test 的 SHAP 值，同样设置 check_additivity 参数为 False
            test_shap_vals[i, :, :] = [shap_values[height_idx][j] for j, height_idx in enumerate(model_prediction)] if (   #将计算得到的 SHAP 值存储在 test_shap_vals 数组中的第 i 个折叠中。如果数据集类别数大于 2 或者模型名称是 'lgbm'，
                    self.get_n_classes() > 2 or self.model_name == 'lgbm') else shap_values         #则将根据 model_prediction 中的索引从 shap_values 中提取对应的 SHAP 值。否则，直接将整个 shap_values 存储在 test_shap_vals 中
            test_preds[i, :, :] = model_preds if self.len_preds > 1 else \
                    model_preds[:, 1].reshape(model_preds.shape[0], 1)     #将每个折叠内的模型预测结果和类别概率结果存储到相应的数组中
            test_all_probas[i, :, :] = model_preds
            #根据指定的评估指标计算每个折叠内的测试分数，并将其存储到相应的数组中
            if self.metric == 'auc':
                fold_score = auc(y_test_np, model_preds, multi_class='ovo') if self.get_n_classes() > 2 else \
                    auc(y_test_np, model_preds[:, 1])
            elif self.metric == 'logloss':
                fold_score = log_loss(y_test_np, model_preds)
            all_test_score[i] = fold_score

        avg_test_score = float('{:.4f}'.format(np.mean(all_test_score, axis=0)))      #计算所有折内的平均测试分数
        print('base test score', str(avg_test_score))   #输出所有折的平均测试分数
        #计算所有折内的预测结果、类别概率和SHAP值的平均值
        avg_test_preds = np.mean(test_preds, axis=0)
        avg_test_all_probas = np.mean(test_all_probas, axis=0)
        avg_test_shap = np.mean(test_shap_vals, axis=0)

        return avg_test_preds, avg_test_shap, avg_test_all_probas

    def predict_second_stage(self, X_test, y_test, config):
        """
         Predict score for the test set using the second-stage model according to config (either P augmented model, SHAP augmented model or P+SHAP augmented model).
         Load the model and calculate prediction and metric value.
        :param X_test: test features
        :param y_test: test target
        :param config: the name of the augmented model to load
        :return: probability predictions
        """
        path = f'models/{self.ds_name}/{self.model_name}/meta_{config}_seed_{self.seed}'
        clf = self.load_model(path)                          #加载模型
        preds = self.predict_proba(clf, (X_test, y_test))    #对数据进行预测
        y_test_np = self.get_y_np(y_test)                    #将y_test转换为np
        #根据规定的性能指标进行计算
        if self.metric == 'auc':
            second_stage_test_score = float('{:.4f}'.format(auc(y_test_np, preds, multi_class='ovo') if self.get_n_classes() > 2 \
                else auc(y_test, preds[:, 1])))
        elif self.metric == 'logloss':
            second_stage_test_score = float('{:.4f}'.format(log_loss(y_test_np, preds)))
        print(f'{config} second stage score', str(second_stage_test_score))
        return preds

    def calc_average_test_score(self, test_preds_base, list_preds, y_test):
        avg_preds = np.mean([test_preds_base] + list_preds, axis=0)
        y_test_np = self.get_y_np(y_test)
        if self.metric == 'auc':
            total_test_score = float('{:.4f}'.format(auc(y_test_np, avg_preds, multi_class='ovo')
                                                     if self.get_n_classes() > 2 else auc(y_test, avg_preds[:, 1])))
        elif self.metric == 'logloss':
            total_test_score = float('{:.4f}'.format(log_loss(y_test_np, avg_preds)))
        return total_test_score

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



