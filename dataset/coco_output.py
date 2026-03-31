"""Utilities for exporting segmentation predictions in COCO JSON format.

Usage example::

    from dataset.coco_output import save_coco_predictions

    categories = [
        {'id': 1, 'name': 'epithelial', 'supercategory': 'nucleus'},
        {'id': 2, 'name': 'inflammatory', 'supercategory': 'nucleus'},
    ]

    results = []
    for image_id, (file_name, height, width, pred_mask) in enumerate(inference_results):
        results.append({
            'image_id': image_id,
            'file_name': file_name,
            'height': height,
            'width': width,
            'pred_mask': pred_mask,   # H x W numpy array, integer class labels
        })

    save_coco_predictions(results, categories, output_path='predictions.json')

The output JSON follows the COCO instance-segmentation format.  Each
connected component of a predicted class is treated as a separate instance
and stored using COCO RLE encoding (``iscrowd=1``).  The file can be
evaluated with the standard COCO API (``pycocotools``).
"""

import json

import numpy as np
from pycocotools import mask as maskUtils
from scipy import ndimage


def mask_to_coco_annotations(pred_mask, image_id, category_map, start_ann_id=1):
    """Convert a semantic segmentation mask to COCO annotation entries.

    Each contiguous connected component for every foreground class is
    recorded as a separate instance annotation.

    Args:
        pred_mask (np.ndarray): ``H x W`` integer array of class labels
            (0 = background, 255 = ignore).
        image_id (int): COCO image ID to associate with these annotations.
        category_map (dict): Mapping ``{label_int: coco_category_id}``.
            Label 0 (background) and 255 (ignore) are skipped automatically.
        start_ann_id (int): Starting annotation ID counter.

    Returns:
        list[dict]: List of COCO annotation dicts ready to be serialised.
    """
    annotations = []
    ann_id = start_ann_id

    for label, cat_id in category_map.items():
        if label in (0, 255):
            continue

        class_mask = (pred_mask == label).astype(np.uint8)
        if class_mask.sum() == 0:
            continue

        # Label individual nuclei / instances via connected-component analysis.
        labeled, num_components = ndimage.label(class_mask)

        for inst_id in range(1, num_components + 1):
            inst_mask = np.asfortranarray(
                (labeled == inst_id).astype(np.uint8)
            )
            area = int(inst_mask.sum())
            if area == 0:
                continue

            # Encode binary mask as COCO RLE.
            rle = maskUtils.encode(inst_mask)
            # 'counts' is bytes; convert to a JSON-serialisable string.
            rle['counts'] = rle['counts'].decode('utf-8')

            # Bounding box in [x, y, width, height] format.
            bbox = [round(float(v), 2) for v in maskUtils.toBbox(rle)]

            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': rle,          # RLE format
                'area': area,
                'bbox': bbox,
                'iscrowd': 1,                 # 1 indicates RLE encoding
            })
            ann_id += 1

    return annotations


def save_coco_predictions(results, categories, output_path):
    """Save segmentation predictions as a COCO-format JSON file.

    Args:
        results (list[dict]): One entry per image.  Each dict must contain:

            * ``'image_id'`` *(int)* – COCO image ID.
            * ``'file_name'`` *(str)* – Image file name (basename).
            * ``'height'`` *(int)* – Image height in pixels.
            * ``'width'`` *(int)* – Image width in pixels.
            * ``'pred_mask'`` *(np.ndarray)* – ``H x W`` integer array of
              predicted class labels (0 = background).

        categories (list[dict]): COCO ``categories`` list, e.g.
            ``[{'id': 1, 'name': 'epithelial', 'supercategory': '...'}, ...]``.
            Categories are expected to appear in the same order as the
            integer class labels produced by the model (label 1 → first
            category, label 2 → second category, etc.).
        output_path (str): Destination path for the output JSON file.
    """
    # Build a label-integer → COCO category_id mapping.
    category_map = {idx + 1: cat['id'] for idx, cat in enumerate(categories)}

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for res in results:
        coco_images.append({
            'id': res['image_id'],
            'file_name': res['file_name'],
            'height': res['height'],
            'width': res['width'],
        })

        anns = mask_to_coco_annotations(
            res['pred_mask'],
            image_id=res['image_id'],
            category_map=category_map,
            start_ann_id=ann_id,
        )
        coco_annotations.extend(anns)
        ann_id += len(anns)

    coco_output = {
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': categories,
    }

    with open(output_path, 'w') as f:
        json.dump(coco_output, f)

    print(
        f'Saved {len(coco_annotations)} annotations for '
        f'{len(coco_images)} images to {output_path}'
    )
