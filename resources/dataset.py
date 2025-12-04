import torchvision.transforms as transforms
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import os


class DatasetSamples(Dataset):
    def __init__(self, dataset, cfg_sample, cfg_train, mode='train'):

        self.dataset = dataset
        self.rgb_max = 2**8
        self.transform = transforms.ToTensor()
        self.samples = []
        last_n_for_test = cfg_train["last_n_for_test"]
        self.gammas = cfg_train["gammas"]

        # 可以留着 target_size 做记录，但后面不再用来裁剪
        self.target_size = cfg_train.get("target_size", (64, 64))

        self.sample_path = os.path.join(
            cfg_sample["output_folder_root"],
            cfg_sample["sample_path"]
        )
        self.context_path = os.path.join(
            cfg_sample["output_folder_root"],
            cfg_sample["context_path"]
        )
        self.target_path = os.path.join(
            cfg_sample["output_folder_root"],
            cfg_sample["target_path"]
        )

        self.samples = [
            f for f in listdir(self.sample_path)
            if isfile(join(self.sample_path, f))
        ]

        if cfg_train["max_samples"] != -1:
            self.samples = self.samples[:cfg_train["max_samples"]]

        # 按 last_n_for_test 做 train / test 切分（文件级）
        if mode == "train":
            if last_n_for_test > 0:
                self.samples = self.samples[:-last_n_for_test]
        else:
            if last_n_for_test > 0:
                self.samples = self.samples[-last_n_for_test:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name = self.samples[idx]

        # sRGB sample patch（H,W,3）~ 0..255
        with open(os.path.join(self.sample_path, file_name), "rb") as f:
            sample = np.load(f, allow_pickle=True)
            sample = np.array(sample, dtype=np.float32) / self.rgb_max

        # sRGB context，用 large_crop 生成，尺寸 cfg_sample["context_size"]
        with open(os.path.join(self.context_path, file_name), "rb") as f:
            context = np.load(f, allow_pickle=True)
            context = np.array(context, dtype=np.float32) / self.rgb_max

        # XYZ target patch（H,W,3），已经在 [0,1]，rggb_max=1.0 只是走一遍兼容逻辑
        with open(os.path.join(self.target_path, file_name), "rb") as f:
            target = np.load(f, allow_pickle=True)
            target = (
                np.clip(
                    np.array(target, dtype=np.float32) - self.dataset["black_level"],
                    0,
                    None,
                )
                / self.dataset["rggb_max"]
            )

        # 🔴 不再做中心裁剪，直接使用完整 patch
        # 采样阶段已经保证 sample/target 同为 (64,64,3)

        sample = self.transform(sample)    # C×H×W
        context = self.transform(context)  # C×Hc×Wc
        target = self.transform(target)    # C×H×W

        # 多 gamma 监督：在通道维上拼接
        targets = []
        for gamma in self.gammas:
            targets.append(torch.pow(target, gamma))  # C×H×W

        targets = torch.cat(targets, dim=0)  # (len(gammas)*C)×H×W

        return sample, targets, context, target
