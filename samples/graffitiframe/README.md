# Graffiti identification

To train:
```
python3 graffiti.py train --dataset=<PATH-TO-DATASET> --weights=<PATH-TO-mask_rcnn_coco.h5>
```

To validate it:
```
python3 graffiti.py batchcsv --weights <PATH-TO-WEIGHTS> --imdir <IMAGESDIR> --outdir <OUTPUTDIR>
```
