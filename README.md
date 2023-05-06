# Invisible-Internal-Defect-Detection
We propose the method of identifying invisible internal defect and reveal their geometric information in structure using object detection (OD) algorithms and surface deformation field acquired from digital image correlation method. Three OD algorithms —Faster RCNN, Mask RCNN, and YOLOv3—were used to investigate the possibility of detecting defect locations and identifying their geometric information.
Faster RCNN was trained using Tensorflow 2.3, Mask RCNN was trained using Tensorflow 2.2 and YOLOv3 was trained using PyTorch 1.8. Python 3.8 was used for all three OD algorithms. 
The source codes of Faster RCNN, Mask RCNN and YOLOv3 were modified to accommodate the input 3D strain tensors save in NumPy format. Further modifications were done to add augmentations and saving all possible output from models to facilitate the post analysis. The results from Faster RCNN and YOLOv3 were saved as csv file, while from Mask RCNN were save as json file. 
The source codes of Faster RCNN, Mask RCNN and YOLOv3 can be found in following sites.
Faster RCNN: https://github.com/rlirli/tf-keras-frcnn
Mask RCNN: https://github.com/ahmedfgad/Mask-RCNN-TF2
YOLOv3: https://github.com/cfotache/pytorch_custom_yolo_training
The train weights of each model are provided along with the annotation files. The strain tensor dataset used in test are also provided along with their original images. 

If you have any questions or comments, please feel free to email: jisikkim@knu.ac.kr or sumanknux@gmail.com
