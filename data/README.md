## Data preparation

### datasets
We use the following datasets for training. 
1. [COCO](http://cocodataset.org/#home)
2. [MPII](http://human-pose.mpi-inf.mpg.de)
3. [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
4. [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
5. [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
6. [AGORA](https://agora.is.tue.mpg.de)
7. [BEDLAM](https://bedlam.is.tue.mpg.de) <br>

The yaml files in `configs` controls which datasets it uses. Please refer to the actual files for more details. <br>
**config.yaml**: use the classic dataset combination from EFT. BEDLAM and AGORA are not included. <br>
**config_bedlam.yaml**: use only synthetic data from BEDLAM and AGORA. The ratio we choose is not from extensive search.
**config_all.yaml**: use synthetic and real data. <br>

After downloading the above datasets from their website, please edit the `data_config.py` file in the main directory. You should fill in the ROOT directory from which you store each dataset. Its organization is consistent with SPIN.


### dataset_extras
We have preprocessed the annotations for the following datasets that are used for training. They are in the same format consistent with SPIN. You can download them from [GoogleDrive](https://drive.google.com/drive/folders/1_yckPZcuEjo0m3UYtvzr9J_c3nLGRiD0?usp=share_link) and put them under `data/dataset_extras`.<br>
The datasets include: **COCO-EFT**, **MPII-EFT**, **MPI-INF-3DHP**, **3DPW**, **AGORA** and **BEDLAM**. <br>
The only annotation we cannot provide is for **H36M** (from MoSH). However, we plan to release proxy fits from the Multi-view ReFit procedure, that hopefully will still boost training when the MoSH fits are not available. 

### pretrained backbone
We use HRNet pretrained on COCO pose esitmation task as weight initialization. The weights can be download for [HRNet-W32](https://drive.google.com/file/d/1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38/view?usp=share_link) and [HRNet-W48](https://drive.google.com/file/d/15T2XqPjW7Ex0uyC1miGVYUv7ULOxIyJI/view?usp=share_link). By default, HRNet-W48 is used as backbone. Please download and put them under `data/pretrain`

### prepare a subset from 3DPW for fast validation
During training, instead of validate on the whole test set, we valid on a pre-cropped subset that cover 1/10 of the whole set. The corresponding annotations are already provided as **3dpw_test_sub.npz** through the GoogleDrive link above with the others. But you will still need to prepare the crops. 
```
python scripts/subsample_3dpw.py
```
Running this script will prepare the crops, save them under the 3DPW root, and generate the corresponding annotation, which is the **3dpw_test_sub.npz** provided.
