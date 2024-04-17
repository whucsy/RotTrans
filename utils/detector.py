import random
import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

loader = transforms.Compose([
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()


# img转tensor
def image_loader(img):
    img = loader(img).unsqueeze(0)
    return img.to(torch.float)


# tensor转img
def tensor_to_PIL(tensor):
    image = tensor.squeeze(0)
    image = unloader(image)
    return image


class Rotation(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being rotated. Default value is 0.5
    """

    def __init__(self, p=0.5, degree=20):
        super().__init__()
        self.p = p
        self.degree = degree

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly rotated image.
        """
        if torch.rand(1) < self.p:
            # degree = random.randint(-(self.degree + 15), self.degree)
            # rot = TF.rotate(img=img, angle=degree)
            # rot = image_loader(rot)
            # predictions = model(rot)
            # boxes = predictions[0]['boxes']
            # if boxes.size(0) > 0:
            #     x1, y1, x2, y2 = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
            #     h = int(y2 - y1)
            #     w = int(x2 - x1)
            #     if h > 50 and w > 50:
            #         box = TF.crop(rot, int(y1), int(x1), h, w)
            #         box = tensor_to_PIL(box)
            #     return box

            degree = random.randint(-self.degree, self.degree)
            rot = TF.rotate(img=img, angle=degree)

            return rot
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
