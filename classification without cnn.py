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


def gaussian(dist, mu=0, sigma=1.0):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.e ** (-0.5*((dist-mu)/sigma)**2)


def data_prepare(path_to_data, appro_period=100, filter_len=2, sampling_rate=10):
    data_ = pd.read_excel(path_to_data).to_numpy()
    print("一共有 " + str(np.shape(data_)[-1]) + "样本")

    # # 数据进行归一化处理 （暂时存在一定问题，先别用）
    # data_only = data_[1::, :]
    # scaler = MinMaxScaler()
    # data_ = np.vstack((data_[0, :], scaler.fit_transform(data_only)))

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

        index_left = peaks[1:-1] - int(period/2)
        index_right = peaks[1:-1] + int(period/2)

        head_ = data_filtered[int(index_left[0]):int(index_right[0])].reshape((1, -1))

        for j in range(len(index_left) - 1):
            # data_slice = np.hstack((label_, one_.reshape(-1, 1)[int(index_left[j+1]):int(index_right[j+1])]))
            head_ = np.vstack((head_, data_filtered[int(index_left[j+1]):int(index_right[j+1])].reshape((1, -1))))

        data_combined_1 = np.hstack((np.ones((len(head_), 1))*label_[i], head_))
        # print(data_combined)

        data_head_all = np.vstack((data_head_all, data_combined_1))
    print(data_head_all)
    return data_head_all[1::, :]


if __name__ == "__main__":
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

    X_nested = from_2d_array_to_nested(data_after[:, 1::])
    X_train, X_test, y_train, y_test = train_test_split(X_nested.iloc[:, [0]], pd.Series(data_after[:, 0].tolist()))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(y_train)
    labels, counts = np.unique(y_train, return_counts=True)
    print(labels, counts)

    # 可视化不同材料的反应曲线
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    for label in labels:
        X_train.loc[y_train == label, 0].iloc[0].plot(ax=ax, label=label)
    plt.legend()
    ax.set(title="Example time series", xlabel="Time")
    plt.show()

    # 利用传统机器学习进行分类

    from sktime.classification.interval_based import RandomIntervalSpectralForest
    risf = RandomIntervalSpectralForest(n_estimators=10)
    risf.fit(X_train, y_train)
    print(risf.score(X_test, y_test))

    from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor

    features = [np.mean, np.std, _slope]
    steps = [
        ("transform", RandomIntervalFeatureExtractor(features=features)),
        ("clf", DecisionTreeClassifier()),
    ]
    tsf2 = Pipeline(steps)
    # tsf2 = ComposableTimeSeriesForestClassifier(estimator=tsf2)
    tsf2.fit(X_train, y_train)
    print(tsf2.score(X_test, y_test))
    y_pred = tsf2.predict(X_test)
    print("打印混淆矩阵")
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(tsf2, X_test, y_test)
    plt.show()
