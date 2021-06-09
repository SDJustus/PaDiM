import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from padim.datasets import LimitedDataset
from padim.utils import propose_regions_cv2 as propose_regions, floating_IoU, get_performance



def test(cfg, padim):
    LIMIT = cfg.test_limit
    TEST_FOLDER = cfg.test_folder
    size = tuple(map(int, cfg.size.split("x")))

    predict_args = {}
    if cfg.compare_all:
        predict_args["compare_all"] = cfg.compare_all

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize( mean=[0.4209137, 0.42091936, 0.42130423],
                            std=[0.34266332, 0.34264612, 0.3432589]),
    ])

    test_dataset = ImageFolder(root=TEST_FOLDER,
                               transform=img_transforms,
                               target_transform=lambda x: int(test_dataset.class_to_idx["0.normal"] == x))

    test_dataloader = DataLoader(batch_size=1,
                                 dataset=LimitedDataset(dataset=test_dataset,
                                                        limit=LIMIT))

    y_trues = []
    y_preds = []

    means, covs, _ = padim.get_params()
    inv_cvars = padim._get_inv_cvars(covs)

    pbar = tqdm(enumerate(test_dataloader))
    for i, test_data in pbar:
        img, y_true = test_data
        res = padim.predict(img, params=(means, inv_cvars), **predict_args)
        if cfg.display:
            amap_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.GaussianBlur(5)
            ])
            amap = res.reshape(1, 1, int(size[0]/4), int(size[1]/4))
            amap = amap_transform(amap)
            
            padim.visualizer.plot_current_anomaly_map(image=img.cpu(), amap=amap.cpu(), train_or_test="test", global_step=i)
        preds = [res.max().item()]

        y_trues.extend(y_true.numpy())
        y_preds.extend(preds)

    # from 1 normal to 1 anomalous
    y_trues = list(map(lambda x: 1.0 - x, y_trues))
    performance = get_performance(y_trues, y_preds)
    print(performance)
    padim.visualizer.plot_performance(1, performance=performance)

    return performance


