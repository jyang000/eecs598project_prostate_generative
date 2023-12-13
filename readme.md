# EECS 598 007 project
Score-based diffusion model for accelerated MRI for prostate imaging

This project is built based on [score-MRI](https://github.com/HJ-harry/score-MRI) and [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch). However the main function, training function, and configurations have been modified. Some comments on the codes are as following
- `main.py` is the main function
- `run.py` contains the function for training
- `datasets.py` is used to load the fastmri prostate dataset (https://github.com/cai2r/fastMRI_prostate) (which needed to be downloaded)
- `inference_real_3d.py` is used to inference the reonstruction of 3D undersampling, the new sampling algorithm is written in `sampling.py`

The pretrained model for the prostate data can be found here: [already trained models]( https://drive.google.com/drive/folders/1623gRCSSLAIodK7bAWZzvAdTgUFPvcnX?usp=sharing).
For which the input image size is `256*256`