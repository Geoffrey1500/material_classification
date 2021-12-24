import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from scipy import stats


def grabTree(filename):
    # 将文件转换为决策树到内存
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def gaussian(dist, mu=0, sigma=1.0):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.e ** (-0.5*((dist-mu)/sigma)**2)


def data_prepare(path_to_data, appro_period=100, filter_len=2, sampling_rate=10):
    print(pd.read_excel(path_to_data).head(6))
    data_ = pd.read_excel(path_to_data).astype(float).to_numpy()*-10**6
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

    period = appro_period*sampling_rate

    data_head_all = np.zeros((1, period))

    for i in range(np.shape(data_)[-1]):
        print("第i个", i)
        # 平滑过滤降采样后数据
        one_ = data_[:, i]# 降采样后数据
        data_filtered = np.convolve(one_, filter_con, 'valid')  # 平滑后

        peaks, _ = find_peaks(data_filtered, distance=appro_period*sampling_rate*0.5, height=np.max(data_filtered)*0.95)
        print("一共有 " + str(len(peaks)) + " 波峰")

        # 波峰位置可视化
        fig1 = plt.figure()
        plt.plot(one_)
        plt.plot(data_filtered)
        plt.plot(peaks, data_filtered[peaks], "x")
        plt.show()

        index_left = peaks[1:-1] - int(period*2/5)
        index_right = peaks[1:-1] + int(period*3/5)

        head_ = data_filtered[int(index_left[0]):int(index_right[0])].reshape((1, -1))

        for j in range(len(index_left) - 1):
            # data_slice = np.hstack((label_, one_.reshape(-1, 1)[int(index_left[j+1]):int(index_right[j+1])]))
            head_ = np.vstack((head_, data_filtered[int(index_left[j+1]):int(index_right[j+1])].reshape((1, -1))))

        # print(data_combined)

        print(data_head_all.shape, head_.shape)

        data_head_all = np.vstack((data_head_all, head_))
    print(data_head_all[1::, :])
    return data_head_all[1::, :]


if __name__ == "__main__":
    collected_data = data_prepare('wood.csv', sampling_rate=5)
    print(collected_data.shape)
    X_nested = from_2d_array_to_nested(collected_data)

    tsf2 = grabTree("classification_tree.pkl")
    y_pred = tsf2.predict(X_nested)
    target_names = ['PC', 'PVC', 'PMMA', '玻璃', '铜', '不锈钢', '铝合金', '铁', '松木', '桐木']

    print(type(y_pred))

    print(target_names[int(stats.mode(y_pred)[0][0])-1])

