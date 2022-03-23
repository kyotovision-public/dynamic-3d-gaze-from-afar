import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from albumentations.core.transforms_interface import DualTransform, to_tuple


def reshape_image_with_aspect(image, bbox, output_size):
    h, w = image.shape[0], image.shape[1]

    original_aspect_ratio = float(h) / w
    output_aspect_ratio = float(output_size[0]) / output_size[1]

    if original_aspect_ratio > output_aspect_ratio: #縦が長すぎる場合
        new_h = int(w * output_aspect_ratio)
        start_index = (h - new_h) // 2 # 基本は真ん中をクロップ

        # bbox を含むようにクロップ
        if start_index > int(bbox[1] * h):
            start_index = int(bbox[1] * h)
        elif start_index + new_h < (bbox[1] + bbox[3]) * h:
            start_index = (bbox[1] + bbox[3]) * h - new_h + 1
            start_index = int(start_index)

        # bbox を切り取った領域に合わせて変更
        bbox[1] = float(bbox[1] * h - start_index) / new_h
        bbox[3] = bbox[3] * (float(h) / new_h)

        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[1] + bbox[3] > 1:
            bbox[3] = 1 - bbox[1]
        if bbox[0] + bbox[2] > 1:
            bbox[2] = 1 - bbox[0]

        image = image[start_index:start_index+new_h, :]
        image = cv2.resize(image, dsize=output_size[::-1])

        return image, bbox
    else:
        new_w = int(h * (1 / output_aspect_ratio))
        start_index = (w - new_w) // 2

        # bbox を含むようにクロップ
        if start_index > int(bbox[0] * w):
            start_index = int(bbox[0] * w)
        elif start_index + new_w < (bbox[0] + bbox[2]) * 2:
            start_index = (bbox[0] + bbox[2]) * w - new_w + 1
            start_index = int(start_index)

        # bbox を切り取った領域に合わせて変更
        bbox[0] = float(bbox[0] * w - start_index) / new_w
        bbox[2] = bbox[2] * (float(w) / new_w)

        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[1] + bbox[3] > 1:
            bbox[3] = 1 - bbox[1]
        if bbox[0] + bbox[2] > 1:
            bbox[2] = 1 - bbox[0]

        image = image[:, start_index:start_index+new_w]
        image = cv2.resize(image, dsize=output_size[::-1])

        return image, bbox


class SingleAttrTransform:
    """
    Superclass for data transformation
    """

    def __init__(self, input_key, output_key):
        self.input_keys = self._validate_key_arg(input_key)
        self.output_keys = self._validate_key_arg(output_key)
        if len(self.input_keys) != len(self.output_keys):
            raise Exception(
                f"len(input_keys) != len(output_keys): {len(self.input_keys)} != {len(self.output_keys)}"
            )

    def __call__(self, item):
        """
        item: dictionary containing each variable in a dataset
        """
        self.before_transform(item)
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            input_seq = item[in_key]
            item[out_key] = self.transform(input_seq)
        return item

    def transform(self, input_seq):
        raise NotImplementedError

    def before_transform(self, item):
        return

    def _validate_key_arg(self, key_or_keys):
        if isinstance(key_or_keys, str):
            return [key_or_keys]
        else:
            return key_or_keys


class RenameAttr(SingleAttrTransform):
    def __init__(self, input_key, output_key):
        super().__init__(input_key, output_key)

    def transform(self, attr):
        return attr


class FilterAttr:
    """
    Keep attributes given by argument, and discard all other arrributes
    """

    def __init__(self, attr_names):
        self.attr_names = attr_names

    def __call__(self, item):
        new_item = {k: v for k, v in item.items() if k in self.attr_names}
        return new_item


class ImageTransform:
    def __init__(self, img_key, transform):
        self.img_key = img_key
        self.transform = transform

    def __call__(self, item):
        item[self.img_key] = self.transform(item[self.img_key])
        return item


class FlipImageAndDirection:
    def __init__(self, img_key="image", direction_key="biternion"):
        self.img_key = img_key
        self.direction_key = direction_key

    def __call__(self, item):
        item[self.img_key] = TF.hflip(item[self.img_key])
        item[self.direction_key][0] *= -1
        return item


######################################
############ Image Transform ############
######################################
class SquarePad:
    def __call__(self, pil_img):
        return TF.center_crop(pil_img, max(*pil_img.size))


class Pad:
    def __init__(self, n):
        self.n = n

    def __call__(self, pil_img):
        w, h = pil_img.size
        return TF.center_crop(pil_img, (h + 2 * self.n, w + 2 * self.n))


class RandomDownsample:
    def __init__(self, r_min=0.2, r_max=0.8):
        self.r_min = r_min
        self.r_max = r_max

    def __call__(self, img):
        """
        img: Tensor with shape (3, h, w)
        """

        r = np.random.uniform(self.r_min, self.r_max)
        original_size = img.shape[1:]
        img = resize(
            resize(
                img, size=(int(original_size[0]*r),
                           int(original_size[1]*r))
            ),
            size=original_size,
        )
        return img


class Denormalize:
    def __init__(self, mean, std, bgr_to_rgb=False):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.bgr_to_rgb = bgr_to_rgb

    def __call__(self, tensor_img):
        out = tensor_img.to("cpu").numpy().transpose(
            (1, 2, 0)) * self.std + self.mean
        if self.bgr_to_rgb:
            out = out[:, :, [2, 1, 0]]

        return out


class RGBToBGR:
    def __call__(self, tensor):
        return tensor[[2, 1, 0], :, :]


######################################
############ Bounding Box ##############
#####################################
class ExpandBB(SingleAttrTransform):
    """
    Expand or shurink the bounding box by multiplying specified arguments
    """

    def __init__(self, t, b, l, r, input_key="bb", output_key=None):
        output_key = output_key or input_key
        super().__init__(input_key, output_key)
        self.t = t
        self.b = b
        self.l = l
        self.r = r

    def transform(self, bb):
        old_w, old_h = bb["w"], bb["h"]
        old_u, old_v = bb["u"], bb["v"]

        lpad = int(old_w * self.l)
        rpad = int(old_w * self.r)
        tpad = int(old_h * self.t)
        bpad = int(old_h * self.b)

        return {
            "w": old_w + lpad + rpad,
            "h": old_h + tpad + bpad,
            "u": old_u - lpad,
            "v": old_v - tpad,
        }


class ExpandBBRect(SingleAttrTransform):
    """
    Make bonding box rectangle.
    """

    def __init__(self, input_key="bb", output_key=None):
        output_key = output_key or input_key
        super().__init__(input_key, output_key)

    def transform(self, bb):
        old_w, old_h = bb["w"], bb["h"]
        old_u, old_v = bb["u"], bb["v"]

        if old_w <= old_h:
            diff = old_h - old_w
            lpad = diff // 2

            return {"w": old_h, "h": old_h, "u": old_u - lpad, "v": old_v}

        if old_h < old_w:
            diff = old_w - old_h
            tpad = diff // 2

            return {"w": old_w, "h": old_w, "u": old_u, "v": old_v - tpad}


class ReshapeBBRect(SingleAttrTransform):
    """
    Crop or Expand the BB tp specified ratio
    """

    def __init__(self, img_ratio, input_key="bb", output_key=None):
        output_key = output_key or input_key
        super().__init__(input_key, output_key)

        assert len(img_ratio) == 2
        self.height = img_ratio[0]
        self.width = img_ratio[1]

    def transform(self, bb):
        old_w, old_h = bb["w"], bb["h"]
        old_u, old_v = bb["u"], bb["v"]

        old_ratio = old_h / old_w
        new_ratio = self.height / self.width

        # 縦が長すぎる場合
        if old_ratio > new_ratio:
            diff = old_h - old_w * (self.height / self.width)
            lpad = diff // 2

            return {"w": old_w, "h": old_h - diff, "u": old_u, "v": old_v + lpad}

        # 横が長すぎる場合
        else:
            diff = old_w - old_h * (self.width / self.height)
            lpad = diff // 2

            return {"w": old_w - diff, "h": old_h, "u": old_u + lpad, "v": old_v}


class CropBB:
    def __init__(self, img_key="image", bb_key="bb", out_key="image"):
        self.img_key = img_key
        self.bb_key = bb_key
        self.out_key = out_key

    def __call__(self, item):
        # self._check_keys(item)
        bb = item[self.bb_key]
        item[self.out_key] = TF.crop(
            item[self.img_key], top=bb["v"], left=bb["u"], height=bb["h"], width=bb["w"]
        )
        return item


class KeypointsToBB:
    def __init__(self, kp_indices):
        if hasattr(kp_indices, "__iter__"):
            kp_indices = list(kp_indices)
        self.kp_indices = kp_indices

    def __call__(self, item):
        out = {k: v for k, v in item.items()}
        kp = item["keypoints"]

        kp = kp[self.kp_indices]
        kp = kp[np.all(kp != 0, axis=1), :]
        u, v = np.min(kp.astype(np.int), axis=0)
        umax, vmax = np.max(kp.astype(np.int), axis=0)
        out["bb"] = {"u": u, "v": v, "w": umax - u, "h": vmax - v}
        return out


class CropWithAspect:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[1:3]

        original_aspect_ratio = h / w
        output_aspect_ratio = self.output_size[0] / self.output_size[1]

        if original_aspect_ratio > output_aspect_ratio:  # もとが縦長すぎる場合
            new_h = int(w * output_aspect_ratio)
            start_index = int((h - new_h) / 2)
            cropped_image = image[:, start_index:(start_index+new_h)]
        else:
            new_w = int(h * (1 / output_aspect_ratio))
            start_index = int((w - new_w) / 2)
            cropped_image = image[:, :, start_index:(start_index+new_w)]

        return cropped_image
