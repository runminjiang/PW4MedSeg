# Enhancing Weakly Supervised 3D Medical Image Segmentation through Probabilistic-aware Learning



## Abstract:

3D medical image segmentation is a challenging task  with crucial implications for disease diagnosis and treatment planning. Recent advances in deep learning have  significantly enhanced fully supervised medical image segmentation. However, this approach heavily relies on labor-intensive and time-consuming fully annotated ground-truth  labels, particularly for 3D volumes. To overcome this limitation, we propose a novel probabilistic-aware weakly supervised learning pipeline, specifically designed for 3D  medical imaging. Our pipeline integrates three innovative  components: a Probability-based Pseudo Label Generation technique for synthesizing dense segmentation masks  from sparse annotations, a Probabilistic Multi-head Self-Attention network for robust feature extraction within our  Probabilistic Transformer Network, and a Probability- informed Segmentation Loss Function to enhance training  with annotation confidence. Demonstrating significant advances, our approach not only rivals the performance of  fully supervised methods but also surpasses existing weakly  supervised methods in CT and MRI datasets, achieving up  to 18.1% improvement in Dice scores for certain organs.

## Contribution
![Alt text](/images/.pdf)

- **Probabilistic-aware Framework**: We introduce a novel probabilistic-aware weakly supervised learning pipeline. Through a comprehensive series of tests, we demonstrate that our method not only significantly enhances performance compared to state-of-the-art weakly supervised methods but also achieves results comparable to fully supervised approaches, highlighting its substantial real-world applicability.

- **Probability-based Pseudo Label Generation**: Within the framework, we innovate by converting sparse 3D point labels into comprehensive dense annotations, leveraging principles from the uncertainty model. This innovative approach minimizes the typical information loss associated with weak labels and enhances segmentation accuracy. Additionally, we simulated the diversity of real-world raw data to test the practicality of our method and achieved promising results.

- **Probabilistic Multi-head Self-Attention (PMSA)**: A critical component of our probabilistic transformer network, it effectively addresses the inherent class variance and noise found in pseudo labels. It plays a pivotal role in enhancing segmentation performance by capturing and utilizing the probabilistic distributions of input-output mappings.

- **Probability-informed Segmentation Loss Function**: To complement the framework, we introduce a novel loss function that incorporates the annotator's confidence level. This loss function aligns the segmentation process more closely with actual boundaries and captures the probabilistic nature of the segmentation task. It also plays a crucial role in reducing the bias in confidence allocation during model training.


## 1. Installation
**Please note the version of monai**
```
git clone https://github.com/runminjiang/PW3MedSeg.git

cd PW3MedSeg

conda create -n PW3MedSeg python==3.9

conda activate PW3MedSeg

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

## 2. Data pre-processing 

**CT**

1. Download the BTCV Dataset [[Google Drive]](https://drive.google.com/drive/folders/19eClJK_E8-lg6L3xbZRw57DfS5Hadupz?usp=sharing) [[百度网盘]](https://pan.baidu.com/share/init?surl=5JepXpVaSTobUQsMo8HFvg&pwd=qdy7) and place it in the `./dataset/btcv` directory.
2. Converting anisotropic images to isotropic images by`python ./Data_preprocessing/anisotropic2isotropic.py`.You may need to use `dataset` to specify which dataset.
3. Convert multi-label image to single-label image by`python ./Data_preprocessing/multi2one.py`.You may need to use `dataset` to specify which dataset and `label_num` to specify which organ. If you are using a custom dataset, you will need to include a custom `label_dict`.

**MRI**

1. Download the Chaos Dataset [[Google Drive]](https://drive.google.com/drive/folders/124zWWxgwS5i972bRgqMGwDp3Fqi1gTn8?usp=drive_link) [[百度网盘]](https://pan.baidu.com/s/1TJUAxJV3iu9joIDbnM8kdw?pwd=qxtx) and place it in the `./dataset/chaos` directory.
2. Converting anisotropic images to isotropic images by`python ./Data_preprocessing/anisotropic2isotropic.py`.You may need to use `dataset` to specify which dataset.
3. Convert multi-label image to single-label image by`python ./Data_preprocessing/multi2one.py`.You may need to use `dataset` to specify which dataset and `label_num` to specify which organ. If you are using a custom dataset, you will need to include a custom `label_dict`.


## 3. Pseudo-label generation

1. run `python ./Data_preprocessing/point2gau.py`. You may need to specify which dataset or organ to use.
2. run `python ./Data_preprocessing/label_thr.py`. You may need to specify which dataset or organ to use.

## 4. Preparing data files
Check out the example file in `./dataset/dataset_0`, you just need to change the train and val sections to your data.

## 5. Train
```
python main.py --logdir btcv_spleen --json_list dataset_spleen_10_200.json --gpu 0 --max_epochs 3500 --use_normal_dataset
```
The training can be done by just modifying the above mentioned parameters and the results are saved to `./run/logdir`.

## 6. Test
```
python test.py --pretrained_dir ./runs/btcv_spleen --json_list dataset_spleen_10_200.json --gpu 0
```
Tests will get test data for DICE and HD95 metrics.
