

<div align="center">

<samp>

<h2> Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery </h1>

<h4> Long Bai*, Mobarakol Islam*, Lalithkumar Seenivasan, and Hongliang Ren </h3>

</samp>   

| **[[```arXiv```](<https://arxiv.org/abs/2305.11692>)]** | **[[```Paper```](<https://ieeexplore.ieee.org/iel7/10160211/10160212/10160403.pdf>)]** |
|:-------------------:|:-------------------:|
    
IEEE International Conference on Robotics and Automation (ICRA) 2023

</div>     
    


If you find our code or paper useful, please cite as

```bibtex
@INPROCEEDINGS{bai2023surgical,
  author={Bai, Long and Islam, Mobarakol and Seenivasan, Lalithkumar and Ren, Hongliang},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Surgical-VQLA:Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery}, 
  year={2023},
  pages={6859-6865},
  doi={10.1109/ICRA48891.2023.10160403}}
```

---
## Abstract
Despite the availability of computer-aided simulators and recorded videos of surgical procedures, junior residents still heavily rely on experts to answer their queries. However, expert surgeons are often overloaded with clinical and academic workloads and limit their time in answering. For this purpose, we develop a surgical question-answering system to facilitate robot-assisted surgical scene and activity understanding from recorded videos. Most of the existing visual question answering (VQA) methods require an object detector and regions based feature extractor to extract visual features and fuse them with the embedded text of the question for answer generation. However, (i) surgical object detection model is scarce due to smaller datasets and lack of bounding box annotation; (ii) current fusion strategy of heterogeneous modalities like text and image is naive; (iii) the localized answering is missing, which is crucial in complex surgical scenarios. In this paper, we propose Visual Question Localized-Answering in Robotic Surgery (Surgical-VQLA) to localize the specific surgical area during the answer prediction. To deal with the fusion of the heterogeneous modalities, we design gated vision-language embedding (GVLE) to build input patches for the Language Vision Transformer (LViT) to predict the answer. To get localization, we add the detection head in parallel with the prediction head of the LViT. We also integrate generalized intersection over union (GIoU) loss to boost localization performance by preserving the accuracy of the question-answering model. We annotate two datasets of VQLA by utilizing publicly available surgical videos from EndoVis-17 and 18 of the MICCAI challenges. Our validation results suggest that Surgical-VQLA can better understand the surgical scene and localized the specific area related to the question-answering. GVLE presents an efficient language-vision embedding technique by showing superior performance over the existing benchmarks.  

<p align="center">
<img src="figures/svqla.png" alt="SurgicalVLQA" width="1000"/>
</p>


---
## Environment

- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py

## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `EndoVis-18-VQLA/` : seq_{1,2,3,4,5,6,7,9,10,11,12,14,15,16}. Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from EndoVIS18 challange.
            - `vqla`
                - `label`: Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                    - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
        - `....`
        - `seq_16`
    - `EndoVis-17-VQLA/` : selected 97 frames from EndoVIS17 challange for external validation. 
        - `left_frames`
        - `vqla`
            - `label`: Q&A pairs and bounding box label.
            - `img_features`: Contains img_features extracted from each frame with different patch size.
                - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
    - `featre_extraction/`:
        - `feature_extraction_EndoVis18-VQLA-frcnn.py`: Used to extract features with Fast-RCNN and ResNet101.
        - `feature_extraction_EndoVis18-VQLA-resnet`: Used to extract features with ResNet18 (based on patch size).
- `models/`: 
    - GatedLanguageVisualEmbedding.py : GLVE module for visual and word embeddings and fusion.
    - LViTPrediction.py : our proposed LViT model for VQLA task.
    - VisualBertResMLP.py : VisualBERT ResMLP encoder from Surgical-VQA.
    - visualBertPrediction.py : VisualBert encoder-based model for VQLA task.
    - VisualBertResMLPPrediction.py : VisualBert ResMLP encoder-based model for VQLA task.
- dataloader.py
- train.py
- utils.py

---
## Dataset
1. EndoVis-18-VQA (Image frames can be downloaded directly from EndoVis Challenge Website)
    - [VQLA](https://drive.google.com/file/d/1m7CSNY9PcUoCAUO_DoppDCi_l2L2RiFN/view?usp=sharing)
2. EndoVis-17-VLQA (External Validation Set)
    - [Images & VQLA](https://drive.google.com/file/d/1PQ-SDxwiNXs5nmV7PuBgBUlfaRRQaQAU/view?usp=sharing)  

---

### Run training
- Train on EndoVis-18-VLQA 
    ```bash
    python train.py --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver lvit --batch_size 64 --epochs 80
    ```

---
## Evaluation
- Evaluate on both EndoVis-18-VLQA & EndoVis-17-VLQA
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
