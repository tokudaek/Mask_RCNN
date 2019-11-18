# Graffiti identification

Preliminaries:
```
conda deactivate; conda create -yn graff tensorflow-gpu scikit-image matplotlib imgaug ipython keras
```

To train:
```
python3 graffiti.py train --dataset=<PATH-TO-DATASET> --weights=<PATH-TO-mask_rcnn_coco.h5>
```

To validate it:
```
python3 graffiti.py batchimg --weights <PATH-TO-WEIGHTS> --imdir ../../images/graff/ --outdir <OUTPUTDIR>
```
