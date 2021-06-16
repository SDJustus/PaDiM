from typing import Tuple

from torch import Tensor
from torch.nn import Module
from efficientnet_pytorch import EfficientNet


class EfficientNetB5(Module):
    embeddings_size = 104
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
        blocks_to_use = []
        feature_not_using_1 = self.efficientnetb5._blocks[0](x)
        feature_1 = self.efficientnetb5._blocks[1](feature_not_using_1)
        feature_not_using_2 = self.efficientnetb5._blocks[2](feature_1)
        feature_2 = self.efficientnetb5._blocks[3](feature_not_using_2)
        feature_3 = self.efficientnetb5._blocks[4](feature_2)

        return feature_1, feature_2, feature_3
