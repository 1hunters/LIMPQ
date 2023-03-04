# LIMPQ (Mixed-Precision Neural Network Quantization via Learned Layer-wise Importance)

Official implementation for paper "Mixed-Precision Neural Network Quantization via Learned Layer-wise Importance"



![](figures/1.png)

## Abstract

The exponentially large discrete search space in mixed-precision quantization (MPQ) makes it hard to determine the optimal bit-width for each layer. Previous works usually resort to iterative search methods on the training set, which consume hundreds or even thousands of GPU-hours. In this study, we reveal that some unique learnable parameters in quantization, namely the scale factors in the quantizer, can serve as importance indicators of a layer, reflecting the contribution of that layer to the final accuracy at certain bit-widths. These importance indicators naturally perceive the numerical transformation during quantization-aware training, which can precisely provide quantization sensitivity metrics of layers. However, a deep network always contains hundreds of such indicators, and training them one by one would lead to an excessive time cost. To overcome this issue, we propose a joint training scheme that can obtain all indicators at once. It considerably speeds up the indicators training process by parallelizing the original sequential training processes. With these learned importance indicators, we formulate the MPQ search problem as a one-time integer linear programming (ILP) problem. That avoids the iterative search and significantly reduces search time without limiting the bit-width search space. For example, MPQ search on ResNet18 with our indicators takes only 0.06 seconds, which improves time efficiency exponentially compared to iterative search methods. Also, extensive experiments show our approach can achieve SOTA accuracy on ImageNet for far-ranging models with various constraints (e.g., BitOps, compress rate).



## Importance Indicators Pre-training (One-time Training for Importance Derivation)
Firstly, we can pre-train the importance indicators for your models, or you can also use our previous indicators (in quantization_search/indicators/importance_indicators_resnet50.pkl). 

### Pre-train the indicators

```
cd indicators_pretraining && python -m torch.distributed.launch --nproc_per_node={NUM_GPUs} main.py {CONFIG.YAML} 
```

You can find the template YAML configuration file in "indicators_pretraining/config_resnet50.yaml". 

Meanwhile, if you want to use your own PyTorch model, you should add it to the *create_model* function (see indicators_pretraining/model/model.py) and designate it in the YAML configuration file. 

### Some Tips 

Pre-training does not require too many epochs, and even does not rely on the full training set, you can try 3~10 epochs and 50% data. 

### Extract the indicators

You should extract the indicators from the checkpoint after pre-training, since these indicators are ***quantization step-size scale-factors*** —— some learnable PyTorch parameters. This is quite easy, since we can traverse the checkpoint and record all step-size scale factors accordingly. 

In pre-training, the quantization step-size scale-factor for each layer has a specific variable name in the weight/activation quantizer (see indicators_pretraining/quan/quantizer/lsq.py, LINE72). 

For example, for layer "*module.layer2.0.conv2*", its activation and weight indicators are named "*module.layer2.0.conv2.quan_a_fn.s*" and "*module.layer2.0.conv2.quan_w_fn.s*", respectively. That means you can access all indicators with these orderly variable names.  

**The indicator extractor example code is in "indicators_pretraining/importance_extractor.py".** 



## ILP-based MPQ Policy Search

### Search with provided constraints

Our code provides two constraints: BitOPs and model size (compression ratio), and at least one constraint should be enabled. 

Once obtaining the indicators, you can perform constraint search several times using the same indicators with below args: 

| Args   | Description                                                  | Example      |
| ------ | ------------------------------------------------------------ | ------------ |
| --i    | path of the importance indicators obtained by "*importance_extractor.py*" | data/r50.pkl |
| --b    | bit-width candidates                                         | 6 5 4 3 2    |
| --wb   | expected weight bit-width                                    | 3            |
| --ab   | expected activation bit-width                                | 3            |
| --cr   | model compression ratio (CR) constraint, cr=0 means disable this constraint | 12.2         |
| --bops | use BitOPs as a constraint                                   | True/False   |

 As an example, one can use the following command to reproduce the result in our paper:

```
python search.py --model resnet50 --i indicators/importance_indicators_resnet50.pkl --b 6 5 4 3 2 --wb 3 --ab 4 --cr 12.2 --bops True 
```

And you will get a MPQ policy (W3A4 & compress ratio 12.2) immediately: 

```
searched weight bit-widths [6, 3, 4, 5, 4, 4, 5, 4, 4, 5, 4, 3, 4, 3, 3, 3, 4, 4, 3, 4, 4, 3, 4, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
searched act bit-widths [6, 4, 6, 6, 6, 3, 4, 6, 3, 4, 5, 3, 5, 5, 6, 3, 4, 5, 3, 4, 5, 3, 4, 4, 4, 6, 5, 5, 4, 5, 5, 4, 6, 5, 3, 5, 6, 3, 5, 6, 3, 6, 4, 4, 6, 2, 6, 4, 6, 6, 4, 6]
```

To search this policy, our ILP solver took 0.35 seconds on a six-core Intel i7-8700 (@ 3.2 GHz) CPU. 

### Additional constraints

You can easily add other constraints (such as on-device latency), please refer the code.  



## Fine-tuning & Evaluation

#### Fine-tuning

You can use any quantization algorithms to finetune your model. 

In our paper, the quantization algorithm is LSQ, see "*quantization_training*" folder and "*quantization_training/config_resnet50.yaml*" for details. 

Please paste your MPQ policy to the YAML file and use conventional training script to finetune the model. You can start from the above searched ResNet50's MPQ policy (W3A4 & compress ratio 12.2) through an example YAML file: 

```
cd quantization_training && python -m torch.distributed.launch --nproc_per_node={NUM_GPUs} main.py finetune_resnet50_w3a4_12.2compression_ratio.yaml
```

#### Evaluation

You can also evaluate the finetuned model in our paper through:

- Download the weights and the YAML file: 

  Please modify the data path/batch size to the proper values. 

  - ResNet50 (W3A4 & 12.2 compression rate), 76.9% IMAGENET top-1 accuracy
    - Weights: https://drive.google.com/file/d/1D82vJj1BD0YPBPAVBs0z1O-5EaxI7bnl/view?usp=share_link
    - YAML file: https://drive.google.com/file/d/1XFLxz7SAjMXdj0KkdNDlYXR-a319JmPp/view?usp=share_link
  - MobileNet-V1 (W4A8 & Weight-only), 2.08MB
    - Weights: https://drive.google.com/file/d/1VCW32TgxB6SteG0wFKXMQluwbamSW46t/view?usp=share_link
    - YAML file: 
  - MobileNet-V1 (W4A4), 9.68G BitOPS
    - Weights: https://drive.google.com/file/d/1eWq1SfkLzdPt24pMQjthNPoZ32EkcK3K/view?usp=share_link
    - YAML file: 

- Evaluate:

  ```
  cd quantization_training && python -m torch.distributed.launch --nproc_per_node=2 main.py {DOWNLOADED_YAML_FILE.YAML}
  ```

  

## Acknowledgement

The authors would like to thank the following insightful open-source projects & papers, this work cannot be done without all of them:

- LSQ implementation: https://github.com/zhutmost/lsq-net
- PuLP: https://github.com/coin-or/pulp
- Network slimming: https://github.com/Eric-mingjie/network-slimming
- HAWQ: https://github.com/Zhen-Dong/HAWQ
- SAT: https://arxiv.org/pdf/1912.10207



## Citation

```
@inproceedings{tang2022mixed,
  title={Mixed-Precision Neural Network Quantization via Learned Layer-wise Importance},
  author={Tang, Chen and Ouyang, Kai and Wang, Zhi and Zhu, Yifei and Wang, Yaowei and Ji, Wen and Zhu, Wenwu},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

