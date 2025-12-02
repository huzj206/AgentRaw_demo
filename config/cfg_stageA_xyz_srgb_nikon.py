# cfg_stageA_xyz_srgb.py

dataset = {
    "name": "FiveK_Nikon_XYZ_StageA",
    "root": "/data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon",
    "rgb_path": "sRGB_train",   # sRGB 子集
    "raw_path": "xyz_train",    # XYZ 子集 (.npy)
    "rggb_max": 2**14,          # Stage A 用不到，保留字段以兼容
    "black_level": 0,
}

cfg_sample = {
    "output_folder_root": "/data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/stageA_patches_train",

    # sample: sRGB input
    "sample_path": "srgb-sample/",
    # target: XYZ GT
    "target_path": "xyz-target/",
    # context: sRGB context
    "context_path": "srgb-context/",

    "sample_size": (64, 64),
    "delta": 64,
    "n_bins": 10,
    "type": "stratified",      # "random" / "stratified"
    "samples_per_channel": 2,  # 每张图每通道 2 个 patch，上限 3*2=6 个 patch

    "context_size": (128, 128),
    "context_size_scale": 0.9,
}

cfg_train = {
    # 这里先占位，采样脚本不用，但后续 Stage A 训练可以直接复用结构
    "tag": "stageA-xyz-srgb",
    "epochs": 100,
    "batch_size": 32,
    "batch_size_test": 128,
    "lr": 1e-3,
    "lr_scaling": 0.01,
    "restart": 16,
    "hidden_size": 128,
    "depth": 8,
    "target_size": (32, 32),
    "max_samples": -1,
    "save_path": "./outputs/",
    "tensorboard_path": "./runs/run_nikon_train/",
    "last_n_for_test": 5,
    "gammas": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
}
