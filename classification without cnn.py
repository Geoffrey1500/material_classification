import pandas as pd
import numpy as np
from sktime.datatypes._panel._convert import (
    from_2d_array_to_nested,
    from_nested_to_2d_array,
    is_nested_dataframe,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sktime.utils.slope_and_trend import _slope
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns


def gaussian(dist, mu=0, sigma=1.0):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.e ** (-0.5*((dist-mu)/sigma)**2)


def data_prepare(path_to_data, appro_period=100, filter_len=2, sampling_rate=10):
    data_ = pd.read_excel(path_to_data).to_numpy()
    print("一共有 " + str(np.shape(data_)[-1]) + "样本")

    # 数据降采样
    data_ = data_[::int(10 / sampling_rate), :]

    # 构建高斯平滑滤波
    smooth_interval = filter_len * sampling_rate
    filter_con = gaussian(np.arange(-smooth_interval, smooth_interval + 1, 1), sigma=smooth_interval / 4)  # 卷积核

    # 高斯滤波可视化
    fig = plt.figure()
    plt.plot(np.arange(-smooth_interval, smooth_interval + 1, 1), filter_con)
    plt.show()

    label_ = np.arange(1, np.shape(data_)[-1] + 1, 1).reshape((-1, 1))
    period = appro_period*sampling_rate

    data_head_all = np.zeros((1, period + 1))

    for i in range(np.shape(data_)[-1]):
        # 平滑过滤降采样后数据
        one_ = data_[:, i]  # 降采样后数据
        data_filtered = np.convolve(one_, filter_con, 'valid')  # 平滑后

        peaks, _ = find_peaks(data_filtered, distance=appro_period*sampling_rate*0.6, height=np.max(data_filtered)*0.95)
        print("一共有 " + str(len(peaks)) + " 波峰")

        # 波峰位置可视化
        fig1 = plt.figure()
        plt.plot(one_)
        plt.plot(data_filtered)
        plt.plot(peaks, data_filtered[peaks], "x")
        plt.show()

        index_left = peaks[1:-1] - int(period*2/5)
        index_right = peaks[1:-1] + int(period*3/5)

        head_ = data_filtered[int(index_left[1]):int(index_right[1])].reshape((1, -1))

        for j in range(len(index_left) - 2):
            # data_slice = np.hstack((label_, one_.reshape(-1, 1)[int(index_left[j+1]):int(index_right[j+1])]))
            head_ = np.vstack((head_, data_filtered[int(index_left[j+2]):int(index_right[j+2])].reshape((1, -1))))

        data_combined_1 = np.hstack((np.ones((len(head_), 1))*label_[i], head_))
        # print(data_combined)

        data_head_all = np.vstack((data_head_all, data_combined_1))
    print(data_head_all)
    return data_head_all[1::, :]


def storeTree(inputTree, filename):
    # 序列化决策树,存入文件
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    # 将文件转换为决策树到内存
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    # 数据预处理
    if os.path.exists("processed_data.xlsx"):
        print("文件已存在，正在读取")
        data_after = pd.read_excel('processed_data.xlsx').values

    else:
        print("文件不存在，正在预处理")
        data_after = data_prepare("data_cut.xlsx", sampling_rate=5)
        print(data_after)

        data_df = pd.DataFrame(data_after)  # 关键1，将ndarray格式转换为DataFrame

        # 将文件写入excel表格中
        writer = pd.ExcelWriter('processed_data.xlsx')  # 关键2，创建名称为 processed_data 的excel表格
        data_df.to_excel(writer, 'page_1', header=False, index=False,
                         float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到 processed_data 表格的第一页中。若多个文件，可以在page_2中写入
        writer.save()  # 保存

    # 数据集分割
    X_nested = from_2d_array_to_nested(data_after[:, 1::])
    X_train, X_test, y_train, y_test = train_test_split(X_nested.iloc[:, [0]], pd.Series(data_after[:, 0].tolist()),
                                                        test_size=0.3)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    labels, counts = np.unique(y_train, return_counts=True)
    print(labels, counts)

    print("查看X_test", X_test)

    # 可视化不同材料的反应曲线
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    for label in labels:
        X_train.loc[y_train == label, 0].iloc[0].plot(ax=ax, label=label)
    plt.legend()
    ax.set(title="Example time series", xlabel="Time")
    plt.show()

    # 判断决策树是否存在
    try:
        # 读取决策树
        print("决策树存在，正在读取")
        tsf2 = grabTree("classification_tree.pkl")
    except Exception:
        print("决策树不存在，正在训练")
        # 训练决策树进行分类
        from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor

        features = [np.mean, np.std, _slope]
        steps = [
            ("transform", RandomIntervalFeatureExtractor("sqrt", features=features)),
            ("clf", DecisionTreeClassifier(max_depth=6)),
        ]
        tsf2 = Pipeline(steps)
        tsf2.fit(X_train, y_train)
        print(tsf2.score(X_test, y_test))

        test = []
        for i in range(10):
            clf = Pipeline(steps=[
                ("transform", RandomIntervalFeatureExtractor("sqrt", features=features)),
                ("clf", DecisionTreeClassifier(max_depth=i + 1)),
            ])

            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            test.append(score)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 8), dpi=80)
        plt.plot(range(1, 11), test)
        plt.show()

        # 保存决策树
        storeTree(tsf2, "classification_tree.pkl")

    scores = cross_val_score(tsf2, X_train, y_train, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    '''
    === 打印分类报告 ===
    TP(True Positive): 预测为正样本， 实际为正样本（预测正确）
    FP(False Positive): 预测为正样本， 实际为负样本 （预测错误）
    FN(False Negative): 预测为负样本，实际为正样本 （预测错误）
    TN(True Negative): 预测为负样本， 实际为负样本 （预测正确）
    
    精确度(precision) = 正确预测的个数(TP)/预测为正样本的个数(TP+FP)
    召回率(recall) = 正确预测值的个数(TP)/实际为正样本的个数(TP+FN)
    F1值 = 2*精度*召回率/(精度+召回率)
    support 为每个标签的出现次数(权重)
    
    micro avg：计算所有数据中预测正确的值
    macro avg：每个类别指标中的未加权平均值(一列)
    weighted avg：每个类别指标中的加权平均
    '''
    from sklearn.metrics import classification_report
    target_names = ['PC', 'PVC', 'PMMA', '玻璃', '铜', '不锈钢', '铝合金', '铁', '松木', '桐木']
    print(classification_report(y_test, tsf2.predict(X_test), target_names=target_names))

    # === 绘制混淆矩阵：真实值与预测值的对比 ===
    y_pred = tsf2.predict(X_test)
    con_mat = confusion_matrix(y_test, y_pred)

    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)

    # === plot ===
    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm, annot=True, cmap='Blues')

    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # 绘制决策树
    from sklearn import tree
    import graphviz
    dot_data = tree.export_graphviz(tsf2.steps[1][1],
                                    out_file=None,
                                    class_names=['PC', 'PVC', 'PMMA', 'Glass', 'Cuprum', 'SS', 'AL', 'Steel', 'Pine', 'Candlenut'],
                                    filled=True,
                                    rounded=True
                                    )
    graph = graphviz.Source(dot_data)
    graph.render("iris")

    print(tsf2.steps[1][1].feature_importances_)
