#!/usr/bin/env python3
"""Inference in bach mode and save to file in pascal xml format
"""

import argparse
import os, sys, time, random, math
from os.path import join as pjoin
import inspect
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import skimage.io
import pathlib
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
rootdir = './'
sys.path.append(rootdir)
sys.path.append(os.path.join(rootdir, "samples/coco/"))
import coco

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

##########################################################
def visualize_detection(image, f, r, outdir):
    """Visualie detections"""
    info(inspect.stack()[0][3] + '()')
    W = 1280; H = 960
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    visualize.display_bboxes_person(image, r['rois'], r['masks'],
                                    r['class_ids'], CLASS_NAMES, r['scores'],
                                    ax=ax)
    plt.tight_layout()
    print(os.path.join(outdir, f.replace('.jpg', '.png')))
    plt.savefig(os.path.join(outdir, f.replace('.jpg', '.png')))
    plt.close()

##########################################################
def export_txt(image, f, r, outdir):
    """Export annotations. Result coming from the maskrcnn is (y1, x1, y2, x2)
    with origin at the TOP left corner (and not in the bottom left)"""
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, f.replace('.jpg', '.txt'))
    if os.path.exists(outpath): return -1

    bboxes = r['rois']
    scores = r['scores']
    classes =np.array(CLASS_NAMES)[r['class_ids']]

    h, w,_ = image.shape

    content = ''
    for i, cl in enumerate(classes):
        if cl != 'person': continue
        y1, x1, y2, x2 = bboxes[i]
        content += '{} {} {} {} {} {}\n'.format(cl, scores[i], x1, y1, x2, y2)

    with open(outpath, 'w') as fh: fh.write(content)
    return 0

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--choice', default='vis', help='[vis], txt')
    parser.add_argument('--imdir', required=True, help='Test-images directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    modeldir = os.path.join(rootdir, "logs")
    modelpath = os.path.join(rootdir, "mask_rcnn_coco.h5")

    if not os.path.exists(modelpath): utils.download_trained_weights(modelpath)

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=modeldir, config=config)
    model.load_weights(modelpath, by_name=True)

    for f in sorted(os.listdir(args.imdir)):
        if not f.endswith('.jpg'): continue
        image = skimage.io.imread(os.path.join(args.imdir, f))
        results = model.detect([image], verbose=1)
        r = results[0]

        if args.choice == 'vis': visualize_detection(image, f, r, args.outdir)
        elif args.choice == 'txt': export_txt(image, f, r, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))


##########################################################
if __name__ == "__main__":
    main()
