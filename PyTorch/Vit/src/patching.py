import torch
import torch.nn as nn

class Patching(nn.Module):
    def __init__(self, patch_size):
        """ [input]
            - patch_size (int) : パッチの縦の長さ（=横の長さ）
        """
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)
    
    def forward(self, x):
        """ [input]
            - x (torch.Tensor) : 画像データ
                - x.shape = torch.Size([batch_size, channels, image_height, image_width])
        """
        x = self.net(x)
        return x