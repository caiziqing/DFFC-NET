import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.stats import pearsonr
from collections import Counter


k = 9
t = 1
t0 = 0


targets = [1, 2, 3, 4, 5]  # 1:defect, 2:badPaint, 3:heatReflection, 4:post, 5:noise
leixing_array = ['defect', 'badPaint', 'heatReflection', 'post', 'noise']


def classificationJudge(jsonfile, x, y):

    shapes = jsonfile['shapes']
    for shape in shapes:
        points = shape['points']
        x_y_1, x_y_2 = points[0], points[1]

        if x_y_1[0] + x_y_1[1] > x_y_2[0] + x_y_2[1]:
            x_y_1, x_y_2 = x_y_2, x_y_1

        x1, y1 = math.floor(x_y_1[0]), math.floor(x_y_1[1])
        x2, y2 = math.ceil(x_y_2[0]), math.ceil(x_y_2[1])

        if x1 <= x <= x2 and y1 <= y <= y2:
            return shape['label']
    return 'no find!'


def classificationJudge_noise(jsonfile, x, y):

    shapes = jsonfile['shapes']
    for shape in shapes:
        points = shape['points']
        x_y_1, x_y_2 = points[0], points[1]
        if x_y_1[0] + x_y_1[1] > x_y_2[0] + x_y_2[1]:
            x_y_1, x_y_2 = x_y_2, x_y_1
        x1, y1 = math.floor(x_y_1[0]), math.floor(x_y_1[1])
        x2, y2 = math.ceil(x_y_2[0]), math.ceil(x_y_2[1])
        if x1 <= x <= x2 and y1 <= y <= y2:
            if shape['label'] == 'noise':
                return 'noise'
    return 'no find!'


def shijian_caozuo(i, j, frames, img_data):

    return [img_data[i, j, t] for t in range(frames)]


def data_kongjian_frame(data, shijian):
    """Package single-frame spatial data into dictionary format"""
    return {
        'frame': float(shijian),
        'kongjian_data': np.array(data, dtype=float).tolist()
    }


def kongjian_caozuo(i, j, h, w, frames, img_data):
    """Obtain the spatiotemporal data of a pixel (temporal and spatial neighborhood)"""
    min_j = max(0, j - k)
    max_j = min(w, j + k + 1)
    data = []
    t1 = t0
    while t1 < frames:
        spatial_data = [img_data[i, j1, t1] for j1 in range(min_j, max_j)]
        data.append(data_kongjian_frame(spatial_data, t1))
        t1 += t
    return data


def dataToJson_TimeAndTemperature(filename, i, j, classification, target, TimeData, TemperatureData):
    """Generate a complete pixel data dictionary (simulating the original JSON structure)"""
    return {
        'filename': filename,
        'position': {'y': float(i), 'x': float(j)},
        'classification': classification,
        'target': target,
        'TimeData': TimeData,
        'TemperatureData': TemperatureData
    }


def caozuo(datapath, binarypath, binary_label_path, filename):
    """Core Data Extraction Function"""

    img_binary = np.load(binarypath)
    jsonfile = json.load(open(binary_label_path, 'r'))
    img_data = np.load(datapath)  #（h, w, frames）
    H, W = img_binary.shape
    h, w, frames = img_data.shape


    defect = []
    badPaint = []
    heatReflection = []
    post = []
    noise = []


    for i in range(H):
        for j in range(W):
            if img_binary[i, j] == 255:
                leixing = classificationJudge(jsonfile, j, i)
                if leixing != 'no find!':
                    time_data = shijian_caozuo(i, j, frames, img_data)
                    spatial_data = kongjian_caozuo(i, j, h, w, frames, img_data)
                    if leixing == 'defect':
                        defect.append(dataToJson_TimeAndTemperature(
                            filename, i, j, leixing, targets[0], time_data, spatial_data
                        ))
                    elif leixing == 'badPaint':
                        badPaint.append(dataToJson_TimeAndTemperature(
                            filename, i, j, leixing, targets[1], time_data, spatial_data
                        ))
                    elif leixing == 'heatReflection':
                        heatReflection.append(dataToJson_TimeAndTemperature(
                            filename, i, j, leixing, targets[2], time_data, spatial_data
                        ))
                    elif leixing in ['leftPost', 'rightPost']:
                        post.append(dataToJson_TimeAndTemperature(
                            filename, i, j, leixing, targets[3], time_data, spatial_data
                        ))
            else:

                leixing = classificationJudge_noise(jsonfile, j, i)
                if leixing == 'noise':
                    time_data = shijian_caozuo(i, j, frames, img_data)
                    spatial_data = kongjian_caozuo(i, j, h, w, frames, img_data)
                    noise.append(dataToJson_TimeAndTemperature(
                        filename, i, j, leixing, targets[4], time_data, spatial_data
                    ))

    return defect, badPaint, heatReflection, post, noise




# kt0 = 16
# kt1 = 20
# kt2 = 24
# kt3 = 28
# kt4 = 32
# KT = [kt0, kt1, kt2, kt3, kt4]


def scalerData(data):
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    return scaler.transform(data).reshape(-1).tolist()


def get_filename_positionX_positinY(data):
    filename = data['filename']
    position = data['position']
    return filename, position['x'], position['y']


def get_TimeData_TemperatureData_classification(data):
    TimeData = data['TimeData']
    classification = data['classification']
    TemperatureData = []
    for T_x in data['TemperatureData']:
        T_x_shuju = T_x['kongjian_data']
        T_x_shuju_l = len(T_x_shuju)

        T_x_shuju = np.array(T_x_shuju)
        # Complete spatial data to 19 pixels
        if T_x_shuju_l < 19:
            T_x_shuju = np.array(T_x_shuju)
            if classification == 'leftPost':

                temp = T_x_shuju[0]
                T_x_shuju = np.insert(T_x_shuju, 0, [temp] * (19 - T_x_shuju_l))
            else:

                temp = T_x_shuju[-1]
                T_x_shuju = np.append(T_x_shuju, [temp] * (19 - T_x_shuju_l))
        TemperatureData.append(T_x_shuju.tolist())
    return TimeData, TemperatureData, classification


def xiaoyu0(data):
    return [np.where(np.array(d) < 0, 0, d).tolist() for d in data]


def scalerData1(data):
    data = np.array(data)
    d_max = np.max(data)
    return data / d_max if d_max != 0 else np.zeros_like(data)


def gaussian(x, mu, sigma):

    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def get_kongjian_chuli(kongjiandata):

    kj_s = []
    co = []
    M = []
    label = 0
    for i in range(30, 100):
        k_data = np.array(kongjiandata[i])
        kn_max = np.max(k_data)
        kn_maxI = np.argmax(k_data)
        M.append(kn_maxI)


        kn_scaler = scalerData1(k_data)

        #Gaussian weights (fitting variance curve)
        i_1 = i - 29
        sigma = -0.0009 * (i_1 ** 2) + 0.1162 * i_1 + 2.0353
        gaussian_weights = [gaussian(j, 19 // 2, sigma) for j in range(19)]
        gaussian_weights = scalerData1(gaussian_weights)


        corr, _ = pearsonr(gaussian_weights, kn_scaler)
        co.append(corr if not math.isnan(corr) else 0)


        if kn_max == 0:
            t = 0
        else:
            t = (np.sum(k_data) / kn_max) / 19
        kj_s.append(t)

    # Label Determination (Whether It Is a Defect Midpoint)
    label = 1 if sum(1 for m in M if m == 9) > 35 else 0
    co = scalerData1(co).tolist()
    return kj_s, co, label


def get_T_t_and_T_x(T_t, T_x_s):

    T_t_scaler = scalerData(T_t)
    return T_t_scaler, T_x_s


def get_s_data(T_s_scaler):
    return list(T_s_scaler)


def gettarget1(data):
    return 1 if data['target'] == 1 else 0


def gettarget2(data):
    """defect=0, badPaint=1, heatReflection=2, post=3, noise=4"""
    target_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    return target_map[data['target']]


def gettarget3(data):
    """defect/badPaint=1，others=0"""
    return 1 if data['target'] in [1, 2] else 0


def filename_classification_positionX_positionY(filename, classification, positionX, positionY):
    return [filename, classification, positionX, positionY]


def target_T_x_t(f_c_x_y, target1, target2, target3, T_x_t, cov, label):
    return f_c_x_y + [target1, target2, target3] + T_x_t + cov


def writeToCSV(path, data):
    pd.DataFrame(data=data).to_csv(path, index=False, encoding='utf-8')



def data_process():
    # Configuration path (modify according to actual situation)
    read_data_folder_1 = '../../data_BP/'
    read_binary_folder_1 = '../path/binary/'
    read_binary_label_folder_1 = '../path/labelme/'
    save_savedir_folder_1 = '../Data/kongjian_data/'

    global_data_save = []

    for folder_1 in os.listdir(read_data_folder_1):
        read_data_folder_2 = os.path.join(read_data_folder_1, folder_1)
        save_dir_1 = os.path.join(save_savedir_folder_1, folder_1)
        os.makedirs(save_dir_1, exist_ok=True)
        dir_name = folder_1[5:]
        for filename_npy in os.listdir(read_data_folder_2):
            filename = filename_npy[:-4]
            print(f"loading：{folder_1}/{filename}")

            datapath = os.path.join(read_data_folder_2, filename_npy)
            binarypath = os.path.join(
                read_binary_folder_1, folder_1,  f"{filename}.npy"
            )
            binary_label_path = os.path.join(
                read_binary_label_folder_1, folder_1, f"{filename}.json"
            )


            defect, badPaint, heatReflection, post, noise = caozuo(
                datapath, binarypath, binary_label_path, filename
            )
            all_class_data = [defect, badPaint, heatReflection, post, noise]


            save_dir_2 = os.path.join(save_dir_1, filename)
            os.makedirs(save_dir_2, exist_ok=True)


            file_level_data = []
            for idx, class_name in enumerate(leixing_array):
                class_data = all_class_data[idx]
                class_level_data = []


                for data in class_data:

                    filename_img, positionX, positionY = get_filename_positionX_positinY(data)
                    TimeData, TemperatureData, classification = get_TimeData_TemperatureData_classification(data)


                    TemperatureData = xiaoyu0(TemperatureData)
                    TemperatureData_s, cov, label = get_kongjian_chuli(TemperatureData)
                    T_t_scaler, T_x_scaler = get_T_t_and_T_x(TimeData, TemperatureData_s)
                    T_x_t = get_s_data(T_x_scaler)  # Extract spatial features


                    target1 = gettarget1(data)
                    target2 = gettarget2(data)
                    target3 = gettarget3(data)


                    f_c_x_y = filename_classification_positionX_positionY(filename_img, classification, positionX,
                                                                          positionY)
                    full_data = target_T_x_t(f_c_x_y, target1, target2, target3, T_x_t, cov, label)
                    full_data = tuple(full_data)


                    class_level_data.append(full_data)
                    file_level_data.append(full_data)
                    global_data_save.append(full_data)


                class_csv_path = os.path.join(save_dir_2, f"{class_name}.csv")
                writeToCSV(class_csv_path, class_level_data)


            file_csv_path = os.path.join(save_dir_1, f"{filename}.csv")
            writeToCSV(file_csv_path, file_level_data)


        global_csv_path = os.path.join(save_savedir_folder_1, f"{dir_name}.csv")
        writeToCSV(global_csv_path, global_data_save)

    print("The entire process has been completed!")


if __name__ == '__main__':
    data_process()