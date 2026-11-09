import torch


class VideoNorm(object):

    """Apply Norm to Image Pixel on GPU"""

    def __init__(self,
                 mean=[0.5, 0.5, 0.5],
                 std=[1.0, 1.0, 1.0]):

        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        

        def __call__(self, img):
            """
                Args:
                    img: float image tensors, (N, 3, H, W)
                Returns:
                    img: normalize float image tensors.
            """

            if torch.max(img) > 1 and self.mean.max() <= 1:
                img.div_(255.0)
            re = img.sub_(self.mean).div_(self.std)

            return re
        