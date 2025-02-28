import torch
import numpy as np
import cv2
import os

class imgNormalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        img = sample['img']
        score = sample['score']
        img = (img - self.mean) / self.var
        sample = {'img': img, 'score': score}
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample['img']
        score = sample['score']
        img = torch.from_numpy(img).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)
        sample = {
            'img': img,
            'score': score
        }
        return sample


class PETCTDS(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, transform):
        super(PETCTDS, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                if ',' not in line:
                    print(f"Skipping malformed line: {line}")
                    continue
                try:
                    dis, score = line.split(',')
                    score = float(score)
                    dis_files_data.append(dis.strip())
                    score_data.append(score)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} -> {e}")
                    continue
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['img_list'])

    def __getitem__(self, idx):
        img_name = self.data_dict['img_list'][idx]
        img = cv2.imread(os.path.join(self.dis_path, img_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype('float32') / 255
        img = np.transpose(img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]

        sample = {
            'img': img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample