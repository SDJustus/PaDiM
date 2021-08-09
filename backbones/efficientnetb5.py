from typing import Tuple
from numpy.core.shape_base import block

from torch import Tensor
from torch.nn import Module
from efficientnet_pytorch import EfficientNet


class EfficientNetB5(Module):
    embeddings_size = 344
    num_patches = 32 * 32

    def __init__(self) -> None:
        super().__init__()
        self.efficientnetb5 = EfficientNet.from_pretrained('efficientnet-b5')

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the three intermediary layers from the EfficientNetB5
        pre-trained model.
        Params
        ======
            x: Tensor - the input tensor of size (b * c * w * h)
        Returns
        =======
            feature_1: Tensor - the residual from layer 1
            feature_2: Tensor - the residual from layer 2
            feature_3: Tensor - the residual from layer 3
        """
        x = self.efficientnetb5._conv_stem(x)
        x = self.efficientnetb5._bn0(x)
        blocks_to_use = [6, 19, 26]
        features =[]
        for i in range(len(self.efficientnetb5._blocks)):
            x = self.efficientnetb5._blocks[i](x)
            if i in blocks_to_use:
                features.append(x)
                blocks_to_use.pop(0)
            if not blocks_to_use: 
                break
        assert len(features) == 3
        return features