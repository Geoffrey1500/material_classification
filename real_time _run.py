import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from scipy import stats
from scipy.ndimage import gaussian_filter1d


def grabTree(filename):
    # 将文件转换为决策树到内存
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def data_prepare(path_to_data, original_sampling_rate, appro_period=50, filter_len=2, down_sampling_scale=2):

    """
    :param path_to_data: 数据保存路径，注意windows操作系统下的反斜杠
    :param original_sampling_rate: 数据原始采样频率，单位：赫兹 （HZ）
    :param appro_period: 数据预估周期，单位：秒（S)
    :param filter_len: 高斯平滑数据考虑范围（向前或向后的长度），单位：秒（S)
    :param down_sampling_scale: 数据将采样比例
    :return: 返回与训练集所用数据相同格式数据
    """

    print(pd.read_excel(path_to_data).head(6))
    data_ = pd.read_excel(path_to_data).astype(float).to_numpy()
    print("一共有 " + str(np.shape(data_)[-1]) + "样本")

    # 数据降采样
    data_ = data_[::int(down_sampling_scale), :]

    # 计算将采用后数据频率
    downscaled_rate = original_sampling_rate/down_sampling_scale

    # 计算高斯平滑数据点
    smooth_interval = filter_len * downscaled_rate

    # 计算数据周期
    period = appro_period*downscaled_rate

    for i in range(np.shape(data_)[-1]):
        print("第i个", i)
        # 平滑过滤降采样后数据
        one_ = data_[:, i]# 降采样后数据
        data_filtered = gaussian_filter1d(one_, sigma=smooth_interval / 8)  # 平滑后

        peaks, _ = find_peaks(data_filtered, distance=appro_period*downscaled_rate*0.8, height=np.max(data_filtered)*0.95)
        print("一共有 " + str(len(peaks)) + " 波峰")

        # 波峰位置可视化
        fig1 = plt.figure()
        plt.plot(one_)
        plt.plot(data_filtered)
        plt.plot(peaks, data_filtered[peaks], "x")
        plt.show()

        index_left = peaks[0] - int(period*2/5)
        index_right = peaks[0] + int(period*3/5)

        # 这里用的是一个临时处理的方法，我的建议是将波峰前 2/5 的数据补全，这样子训练和预测的数据集能够统一
        print(max(int(index_left), 0), max(int(index_right), period))
        head_ = data_filtered[int(max(int(index_left), 0)):int(max(int(index_right), period))].reshape((1, -1))

    return head_


if __name__ == "__main__":
    collected_data = data_prepare('Cu.csv', original_sampling_rate=10, down_sampling_scale=1)
    print(collected_data.shape)
    X_nested = from_2d_array_to_nested(collected_data)

    tsf2 = grabTree("classification_tree.pkl")
    y_pred = tsf2.predict(X_nested)
    target_names = ['PC', 'PVC', 'PMMA', '玻璃', '铜', '不锈钢', '铝合金', '铁', '松木', '桐木']

    print(type(y_pred))

    print(target_names[int(stats.mode(y_pred)[0][0])-1])
