# Surgical-VQLA

<div align="center">

<samp>

<h2> Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery </h1>

<h4> Long Bai*, Mobarakol Islam*, Lalithkumar Seenivasan and Hongliang Ren </h3>

</samp>   

</div>     
    
---

If you find our code or paper useful, please cite as

```bibtex

```

---
## Abstract
Despite the availability of computer-aided simulators and recorded videos of surgical procedures, junior residents still heavily rely on experts to answer their queries. However, expert surgeons are often overloaded with clinical and academic workloads and limit their time in answering. For this purpose, we develop a surgical question-answering system to facilitate robot-assisted surgical scene and activity understanding from recorded videos. Most of the existing visual question answering (VQA) methods require an object detector and regions based feature extractor to extract visual features and fuse them with the embedded text of the question for answer generation. However, (i) surgical object detection model is scarce due to smaller datasets and lack of bounding box annotation; (ii) current fusion strategy of heterogeneous modalities like text and image is naive; (iii) the localized answering is missing, which is crucial in complex surgical scenarios. In this paper, we propose Visual Question Localized-Answering in Robotic Surgery (Surgical-VQLA) to localize the specific surgical area during the answer prediction. To deal with the fusion of the heterogeneous modalities, we design gated vision-language embedding (GVLE) to build input patches for the Language Vision Transformer (LViT) to predict the answer. To get localization, we add the detection head in parallel with the prediction head of the LViT. We also integrate generalized intersection over union (GIoU) loss to boost localization performance by preserving the accuracy of the question-answering model. We annotate two datasets of VQLA by utilizing publicly available surgical videos from EndoVis-17 and 18 of the MICCAI challenges. Our validation results suggest that Surgical-VQLA can better understand the surgical scene and localized the specific area related to the question-answering. GVLE presents an efficient language-vision embedding technique by showing superior performance over the existing benchmarks.  

<p align="center">
<img src="figures/svqla.png" alt="SurgicalVLQA" width="1000"/>
</p>


---
## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `EndoVis-18-VQA/` : seq_{1,2,3,4,5,6,7,9,10,11,12,14,15,16}. Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from  EndoVIS18 challange.
            - `vqla`
                - `label`: Classification Q&A pairs.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5
        - `....`
        - `seq_16`
    - `featre_extraction/`:
        - `feature_extraction_EndoVis18-VQA-frcnn.py`: Used to extract features with Fast-RCNN and ResNet101.
        - `feature_extraction_EndoVis18-VQA-resnet`: Used to extract features with ResNet18 (based on patch size).
- `models/`: 
    - VisualBertResMLP.py : Our proposed encoder.
    - visualBertClassification.py : VisualBert encoder-based classification model.
    - VisualBertResMLPClassification.py : VisualBert ResMLP encoder-based classification model.
- dataloader.py
- train.py
- utils.py

---
## Dataset (will release dataset after acceptance)
1. EndoVis-18-VQA
    - Images
    - VQLA
2. EndoVis-17-VLQA
    - Images
    - VQLA  

---

### Run training
- Train on EndoVis-18-VLQA 
    ```bash
    python train.py --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver lvit --batch_size 64 --epochs 80
    ```

---
## Evaluation
- Evaluate both on EndoVis-18-VLQA & EndoVis-17-VLQA
    ```bash
    python train.py --validate True --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver lvit --batch_size 64
    ```

---
## References
Code adopted and modified from:
1. VisualBERT model
    - Paper [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557).
    - official pytorch implementation [Code](https://github.com/huggingface/transformers.git).

2. VisualBERT ResMLP model
    - Paper [Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformer](https://arxiv.org/abs/2206.11053).
    - Official Pytorch implementation [Code](https://github.com/lalithjets/Surgical_VQA).

3. DETR
    - Paper [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).
    - Official Pytorch implementation [Code](https://github.com/facebookresearch/detr).

---

## Contact
For any queries, please raise an issue or contact [Long Bai](mailto:b.long@link.cuhk.edu.hk).

---
