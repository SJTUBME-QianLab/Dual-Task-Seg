This repository holds the PyTorch code for the paper

**A Dual-Task Synergy-Driven Generalization Framework for Pancreatic Cancer Segmentation in CT Scans**

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Medical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Author List

Jun Li, Yijue Zhang, Haibo Shi, Minhong Li, Qiwei Li and Xiaohua Qian

# Abstract

Pancreatic cancer, characterized by its notable prevalence and mortality rates, demands accurate lesion delineation for effective diagnosis and therapeutic interventions. The generalizability of extant methods is frequently compromised due to the pronounced variability in imaging and the heterogeneous characteristics of pancreatic lesions, which may mimic normal tissues and exhibit significant inter-patient variability. Thus, we propose a generalization framework that synergizes pixel-level classification and regression tasks, to accurately delineate lesions and improve model stability. This framework not only seeks to align segmentation contours with actual lesions but also uses regression to elucidate spatial relationships between diseased and normal tissues, thereby improving tumor localization and morphological characterization. Enhanced by the reciprocal transformation of task outputs, our approach integrates additional regression supervision within the segmentation context, bolstering the model’s generalization ability from a dual-task perspective. Besides, dual self-supervised learning in feature spaces and output spaces augments the model’s representational capability and stability across different imaging views. Experiments on 594 samples composed of three datasets with significant imaging differences demonstrate that our generalized pancreas segmentation results comparable to mainstream in-domain validation performance (Dice: 84.07%). More importantly, it successfully improves the results of the highly challenging cross-lesion generalized pancreatic cancer segmentation task by 9.51%. Thus, our model constitutes a resilient and efficient foundational technological support for pancreatic disease management and wider medical applications. The codes will be released at https://github.com/SJTUBME-QianLab/Dual-Task-Seg. 

# Main Requirements

Certain necessary dependencies need to be installed to run the code. The main libraries are listed below. For a complete list of environment dependencies, please refer to `requirements.txt`.

- Pyhton 3.8.10
- PyTorch 1.8.1
- Numpy 1.22.3
- Opencv-Python 4.5.5.64
- Scikit-Image 0.21.0
- MedPy 0.4.0
- SimpleITK 2.3.1
- Matplotlib 3.5.2 
- Seaborn 0.11.2


# Training and Testing

Both the training and testing processes can be completed on NVIDIA graphics cards with 24 GB of memory, such as the Nvidia RTX 3090 GPU, Nvidia A30 GPU, and higher-performance GPUs.

## Training

Note that the training of coarse and fine segmentation models for both the pancreas and pancreatic cancer can be conducted in parallel simultaneously, and the training process does not need to be executed in sequence.

Firstly, a simplified pancreas segmentation model needs to be trained to obtain the pancreatic ROI with maximum efficiency.

Navigate to the `train_pancreas` folder using the `cd` command, and run the coarse segmentation training by this command:

```
python train_sh_pancreas_coarse.py
```

Run the fine segmentation training by this command:

```
python train_sh_pancreas_fine.py
```

After obtaining the pancreatic segmentation model, a pancreatic cancer segmentation model needs to be trained.

Navigate to the `train_tumor` folder using the `cd` command, and run the coarse segmentation training by this command:

```
python train_sh_tumor_coarse.py
```

Run the fine segmentation training by this command:

```
python train_sh_tumor_fine.py
```

After training, the weights will be saved in ```./checkpoints``` folder.


## Testing

Note that the testing procedure needs to be strictly executed in the order of coarse pancreas segmentation, fine pancreas segmentation, coarse tumor segmentation, and fine tumor segmentation.

We assign a timestamp for each experiment, so when testing, you need to modify the `time` in the filename to the specific time.

Our comparisons with other methods are all conducted based on the same pancreatic ROI, during both the coarse and fine segmentation stages of the tumor.

Execute the following commands in sequence:

```
cd evaluate_pancreas
python evaluate_sh_pancreas_coarse.py
python evaluate_sh_pancreas_fine.py
```
```
cd ../evaluate_tumor
python evaluate_sh_tumor_coarse.py
python evaluate_sh_tumor_fine.py
```

# Citation

If you find this project useful for your research, please consider citing:

```
@inproceedings{
  title     = {A Dual-Task Synergy-Driven Generalization Framework for Pancreatic Cancer Segmentation in CT Scans},
  author    = {Jun Li, Yijue Zhang, Haibo Shi, Minhong Li, Qiwei Li and Xiaohua Qian},
  month     = {December}，
  year      = {2024},
}
```

# Contact

For any question, feel free to contact

```
Jun Li : dirk.li@outlook.com
```


