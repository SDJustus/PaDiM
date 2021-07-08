from tqdm import tqdm
from torchvision import transforms

from padim.utils import get_performance, write_inference_result
import os


def test(cfg, padim, dataloader):
    size = tuple(map(int, cfg.size.split("x")))

    predict_args = {}
    y_trues = []
    y_preds = []

    means, covs, _ = padim.get_params()
    inv_cvars = padim._get_inv_cvars(covs)

    pbar = tqdm(enumerate(dataloader))
    file_names = []
    for i, test_data in pbar:
        img, y_true, file_name = test_data
        res = padim.predict(img, params=(means, inv_cvars), **predict_args)
        print("res", str(res))
        print(padim.num_patches)
        if cfg.display:
            amap_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.GaussianBlur(5)
            ])
            #if "efficient" in cfg.backbone:
            #    w = int(size[0]/2)
            #    h = int(size[1]/2)
            #else:
            w = int(size[0]/4)
            h = int(size[1]/4)
            amap = res.reshape(1, 1, w, h)
            amap = amap_transform(amap)
            
            padim.visualizer.plot_current_anomaly_map(image=img.cpu(), amap=amap.cpu(), train_or_test="test", global_step=i)
            
        preds = [res.max().item()]

        y_trues.extend(y_true.numpy())
        y_preds.extend(preds)
        file_names.append(file_name)

    # from 1 normal to 1 anomalous
    y_trues = list(map(lambda x: 1.0 - x, y_trues))
    performance, t = get_performance(y_trues, y_preds)
    
    padim.visualizer.plot_pr_curve(y_trues=y_trues, y_preds=y_preds, t=t)
    padim.visualizer.plot_performance(1, performance=performance)
    
    if cfg.inference:
        write_inference_result(file_names=file_names, y_preds=y_preds, y_trues=y_trues, outf=os.path.join(cfg.params_path, "classification_result_" + str(cfg.name) + ".json"))

    return performance


