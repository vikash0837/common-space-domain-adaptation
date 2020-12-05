# common-space-domain-adaptation
## Steps to Run the code
1. you can create Anaconda environment along with all the pre-requisite using "vikash_env.yml" file.
2. For training cycyleGAN please follow the precedure mentioned over "https://github.com/aitorzip/PyTorch-CycleGAN"
3. For cross camera adaptation, go to common space domain adaptation folder
4. Follow the data preparation guidelines availave at "https://github.com/kevinhkhsu/DA_detection"
5. Replace existing folders with your synthetic folders
6. Run the training script
`./experiments/scripts/train_adapt_faster_rcnn_stage2.sh [GPU_ID] [Adapt_mode] vgg16`
# Specify the GPU_ID you want to use
# Adapt_mode selection:
#   'K2C': KITTI->Cityscapes
#   'C2F': Cityscapes->Foggy Cityscapes
#   'C2BDD': Cityscapes->BDD100k_day
# Example:
`./experiments/scripts/train_adapt_faster_rcnn_stage2.sh 0 K2C vgg16`

7. Similarly for testing, run the test scripts
./experiments/scripts/test_adapt_faster_rcnn_stage2.sh [GPU_ID] [Adapt_mode] vgg16
# Specify the GPU_ID you want to use
# Adapt_mode selection:
#   'K2C': KITTI->Cityscapes
#   'C2F': Cityscapes->Foggy Cityscapes
#   'C2BDD': Cityscapes->BDD100k_day
# Example:
`./experiments/scripts/test_adapt_faster_rcnn_stage2.sh 0 K2C vgg16`

8. For foggy experiment, please go to "foggy_exoeriment" and follow from sept 1 to 7
9. for AADA experiment, please go to "foggy_experiement/AADA" and rperform the experiment
