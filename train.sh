
python /projects/shart/digital_pathology/scripts/General-OneShot/train.py  \
-t /projects/shart/digital_pathology/data/dogs_vs_cats/train \
-v /projects/shart/digital_pathology/data/dogs_vs_cats/val \
-m ResNet50 \
-o RMSprop \
-p 256 \
-l /projects/shart/digital_pathology/data/dogs_vs_cats/log -r 0.001 \
-c 128 -e 10 -b 32  -V DEBUG --filetype images
#-L BinaryCrossentropy
