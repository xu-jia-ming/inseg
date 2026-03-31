"""COCO-format segmentation dataset for incremental learning.

Expected directory layout::

    root/
        annotations/
            instances_train.json
            instances_val.json
        images/
            train/
                <image files>
            val/
                <image files>

The annotation JSON must follow the COCO instance-segmentation format.
Each annotation's ``category_id`` is mapped to an integer class label
(1-indexed; 0 is reserved for background) in the order that categories
appear in the JSON ``categories`` list.
"""

import copy
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from pycocotools.coco import COCO
from torch import distributed

from dataset import transform
from .utils import Subset, filter_images, group_images


class CocoSegmentation(data.Dataset):
    """Per-pixel semantic segmentation dataset built from COCO annotations.

    Args:
        root (str): Root directory that contains ``annotations/`` and
            ``images/`` sub-directories.
        image_set (str): Split to load – ``'train'`` or ``'val'``.
        is_aug (bool): Unused; kept for API compatibility with other
            dataset classes in this project.
        transform (callable, optional): Joint image/target transform
            applied to ``(PIL.Image, PIL.Image)`` pairs.
    """

    def __init__(self, root, image_set='train', is_aug=True, transform=None):
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform

        ann_file = os.path.join(
            self.root, 'annotations', f'instances_{image_set}.json'
        )
        if not os.path.exists(ann_file):
            raise RuntimeError(
                f'Annotation file not found: {ann_file}\n'
                'Expected COCO-format JSON at '
                'root/annotations/instances_{train|val}.json'
            )

        self.coco = COCO(ann_file)
        self.img_ids = sorted(self.coco.imgs.keys())

        # Map COCO category_id -> sequential class label (1-indexed).
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_label = {
            cat['id']: idx + 1 for idx, cat in enumerate(cats)
        }
        self.classes = {0: 'background'}
        self.classes.update({idx + 1: cat['name'] for idx, cat in enumerate(cats)})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _img_path(self, file_name):
        return os.path.join(self.root, 'images', self.image_set, file_name)

    def _load_mask(self, img_id, height, width):
        """Return a ``PIL.Image`` with per-pixel integer class labels."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            label = self.cat_id_to_label.get(ann['category_id'], 0)
            inst_mask = self.coco.annToMask(ann)
            mask[inst_mask > 0] = label
        return Image.fromarray(mask)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        img = Image.open(self._img_path(img_info['file_name'])).convert('RGB')
        target = self._load_mask(
            img_id, img_info['height'], img_info['width']
        )

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, img_info['file_name']

    def viz_getter(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = self._img_path(img_info['file_name'])

        raw_image = Image.open(image_path).convert('RGB')
        target = self._load_mask(
            img_id, img_info['height'], img_info['width']
        )
        if self.transform is not None:
            img, target = self.transform(copy.deepcopy(raw_image), target)
        else:
            img = copy.deepcopy(raw_image)
        return image_path, raw_image, img, target


class CocoSegmentationIncremental(data.Dataset):
    """Incremental-learning wrapper around :class:`CocoSegmentation`.

    Follows the same API as ``MoNuSACSegmentationIncremental`` and
    ``CoNSePSegmentationIncremental``.

    Args:
        root (str): Dataset root directory.
        train (bool): Load ``'train'`` split when ``True``, else ``'val'``.
        transform (callable, optional): Joint image/target transform.
        labels (list of int): Class labels for the *current* step.
        labels_old (list of int): Class labels from all *previous* steps.
        idxs_path (str, optional): Path to cache the filtered index array.
        masking (bool): If ``True`` mask out labels not relevant to the
            current step.
        overlap (bool): If ``True`` keep images that contain any label in
            ``labels`` even if they also contain unknown labels.
        data_masking (str): One of ``'current'``, ``'current+old'``,
            or ``'new'``.
        test_on_val (bool): Split training set 80/20 into train/val when
            no dedicated val split is desired.
        image_set (str, optional): Explicit split name; overrides ``train``.
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        masking=True,
        overlap=True,
        data_masking='current',
        test_on_val=False,
        image_set=None,
        **kwargs,
    ):
        split = image_set if image_set is not None else ('train' if train else 'val')
        full_coco = CocoSegmentation(root, image_set=split, is_aug=True, transform=None)

        self.labels = []
        self.labels_old = []

        if labels is not None:
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), (
                'labels and labels_old must be disjoint sets'
            )

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old
            self.order = [0] + labels_old + labels

            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_coco, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if test_on_val:
                rnd = np.random.RandomState(1)
                rnd.shuffle(idxs)
                train_len = int(0.8 * len(idxs))
                idxs = idxs[:train_len] if train else idxs[train_len:]

            masking_value = 0
            self.inverted_order = {
                label: self.order.index(label) for label in self.order
            }
            self.inverted_order[255] = 255

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x]
                    if x in self.inverted_order
                    else masking_value
                )
            )

            if masking:
                if data_masking == 'current':
                    tmp_labels = self.labels + [255]
                    target_transform = reorder_transform
                elif data_masking == 'current+old':
                    tmp_labels = labels_old + self.labels + [255]
                    target_transform = reorder_transform
                elif data_masking == 'all':
                    raise NotImplementedError(
                        f'data_masking={data_masking} not yet implemented.'
                    )
                elif data_masking == 'new':
                    tmp_labels = self.labels
                    masking_value = 255
                    target_transform = tv.transforms.Lambda(
                        lambda t: t.apply_(
                            lambda x: self.inverted_order[x]
                            if x in tmp_labels
                            else masking_value
                        )
                    )
                else:
                    assert False, f'Unknown data_masking value: {data_masking}'

                self.dataset = Subset(full_coco, idxs, transform, target_transform)
            else:
                self.dataset = full_coco
        else:
            self.dataset = full_coco

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
