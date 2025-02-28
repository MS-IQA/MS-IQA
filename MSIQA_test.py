import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from torchvision import transforms
from utils.utils import PETCTDS, imgNormalize, ToTensor
import models.MSIQA_NET as MSIQA_NET

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def eval_epoch(net, test_loader):
    with torch.no_grad():
        net.eval()
        pred_epoch, labels_epoch = [], []

        for data in tqdm(test_loader):
            x = data['img'].cuda()
            labels = torch.squeeze(data['score'].type(torch.FloatTensor)).cuda()
            pred = net(x)
            pred_epoch = np.append(pred_epoch, pred.data.cpu().numpy())
            labels_epoch = np.append(labels_epoch, labels.data.cpu().numpy())

        rho_s, _ = map(abs, spearmanr(pred_epoch, labels_epoch))
        rho_p, _ = map(abs, pearsonr(pred_epoch, labels_epoch))

        print(
            f'Test Results: |SROCC|: {rho_s:.4f}, |PLCC|: {rho_p:.4f}')


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


    print("Starting Evaluation...")

    test_path = "./PET-CT-IQA-DS/test/"
    test_label_path = "./PET-CT-IQA-DS/label.txt"
    model_weight_path = "./model_weights/MSIQA.pt"

    test_dataset = PETCTDS(
        dis_path=test_path,
        txt_file_name=test_label_path,
        transform=transforms.Compose([imgNormalize(0.5, 0.5), ToTensor()]),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=16,
        num_workers=8
    )

    # load model
    net = MSIQA_NET.MSIQA_NET()
    net = net.cuda()

    # load trained weights
    state_dict = torch.load(model_weight_path, map_location='cuda')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # Evaluation
    eval_epoch(net, test_loader)
