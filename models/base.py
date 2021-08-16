from typing import Union, Tuple

import torch
from torch import Tensor, device as Device

from utils import embeddings_concat
from backbones import ResNet18,ResNet50, WideResNet50, EfficientNetB5
from utils.visualizer import Visualizer


class BaseModel:
    """The embedding backbone shared by PaDiM and PaDiMSVDD
    """

    def __init__(self, num_embeddings: int, device: Union[str, Device],
                 backbone: str, cfg=None):
        self.device = device
        self.num_embeddings = num_embeddings
        self.visualizer = Visualizer(cfg)
        self.size = cfg.size
        self.cfg=cfg
        print(self.size)

        if self.size is not None:
            self._init_backbone_with_size(backbone, self.size)
        else:
            self._init_backbone(backbone)

        self.embedding_ids = torch.randperm(
            self.max_embeddings_size)[:self.num_embeddings].to(self.device)

    def _get_backbone(self):
        if isinstance(self.model, ResNet18):
            backbone = "resnet18"
        elif isinstance(self.model, ResNet50):
            backbone = "resnet50"
        elif isinstance(self.model, WideResNet50):
            backbone = "wide_resnet50"
        elif isinstance(self.model, EfficientNetB5):
            backbone = "efficientnetb5"
        else:
            raise NotImplementedError(str(self.model) + " " + str(ResNet18) + " " + str(isinstance(self.model, ResNet18)))

        return backbone

    def _init_backbone_with_size(self, backbone: str, size: Tuple[int, int]) -> None:
        self._init_backbone(backbone)
        empty_batch = torch.zeros((1, 3) + size, device=self.device)
        feature_1, _, _ = self.model(empty_batch)
        _, _, w, h = feature_1.shape
        self.num_patches = w * h
        self.model.num_patches = w * h
        

    def _init_backbone(self, backbone: str) -> None:
        if backbone == "resnet18":
            self.model = ResNet18().to(self.device)
        elif backbone == "resnet50":
            self.model = ResNet50().to(self.device)
        elif backbone == "wide_resnet50":
            self.model = WideResNet50().to(self.device)
        elif backbone == "efficientnetb5":
            self.model = EfficientNetB5().to(self.device)
        else:
            raise Exception(f"unknown backbone {backbone}, "
                            "choose one of ['resnet18', 'resnet50', 'wide_resnet50', 'efficientnetb5]")

        self.num_patches = self.model.num_patches
        self.max_embeddings_size = self.model.embeddings_size
        

    def _embed_batch(self, imgs: Tensor, with_grad: bool = False) -> Tensor:
        self.model.eval()
        with torch.set_grad_enabled(with_grad):
            feature_1, feature_2, feature_3 = self.model(imgs.to(self.device))
            
        embeddings = embeddings_concat(feature_1, feature_2)
        embeddings = embeddings_concat(embeddings, feature_3)
        #print("b4",embeddings.shape)
        embeddings = torch.index_select(
            embeddings,
            dim=1,
            index=self.embedding_ids,
        )
        #print("after", embeddings.shape)
        return embeddings

    def _embed_batch_flatten(self, imgs, *args):
        embeddings = self._embed_batch(imgs, *args)
        _, C, _, _ = embeddings.shape
        return embeddings.permute(0, 2, 3, 1).reshape((-1, C))
