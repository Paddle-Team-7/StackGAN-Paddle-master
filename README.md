# StackGAN-paddle

Paddle implementation for reproducing COCO results in the paper [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v2.pdf) by Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas. The network structure is slightly different from the tensorflow implementation. 

<img src="examples/framework.jpg" width="850px" height="370px"/>


### Dependencies
python 3.7

Paddle


**Reproduction**

We have trained the two stage models. But we can not get evaluate datasets, so we can't get results.

**Data**

1. Download our preprocessed char-CNN-RNN text embeddings for [training coco](https://drive.google.com/open?id=0B3y_msrWZaXLQXVzOENCY2E3TlU) and  [evaluating coco](https://drive.google.com/open?id=0B3y_msrWZaXLeEs5MTg0RC1fa0U), save them to `data/coco`.
  - [Optional] Follow the instructions [reedscot/icml2016](https://github.com/reedscot/icml2016) to download the pretrained char-CNN-RNN text encoders and extract text embeddings.
2. Download the [coco](http://cocodataset.org/#download) image data. Extract them to `data/coco/`.



**Training**
- The steps to train a StackGAN model on the COCO dataset using our preprocessed embeddings.
  - Step 1: train Stage-I GAN (e.g., for 120 epochs) `python main.py --cfg cfg/coco_s1.yml`
  - Step 2: train Stage-II GAN (e.g., for another 120 epochs) `python main.py --cfg cfg/coco_s2.yml`
- `*.yml` files are example configuration files for training/evaluating our models.



**Pretrained Model**
- [StackGAN for coco](https://pan.baidu.com/s/13wgCoESnQ41PqYVJ3PGnqQ) (dlga). Download and save it to `models/coco`.


**Evaluating**
- Run `python main.py --cfg cfg/coco_eval.yml`

