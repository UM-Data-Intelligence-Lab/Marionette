<img width="1277" alt="image" src="https://github.com/user-attachments/assets/e694003c-c284-4ba3-bfe8-ddbdcfeda755" /># Marionette
Marionette offers fine-grained controllable generative human trajectory data modeling with both global and partial mobility-related contexts. Please see the details in our paper below:  
- Bangchao Deng, Ling Ding, Lianhua Ji, Chunhua Chen, Xin Jing, Bingqing Qu, Dingqi Yang*, Marionette: Fine-Grained Conditional Generative Modeling of Spatiotemporal Human Trajectory Data Beyond Imitation, In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'25), Aug. 2025, Toronto.

## Requirements
```
python: 3.10 
torch: 2.0.0  
recbole: 1.2.1 
numpy: 1.26.4
pandas: 2.2.2
pytorch-lightning: 1.9.5
hydra-core
wandb
torchtyping
matplotlib
einops
rich
GPU with CUDA 11.7
```

### Training, Sampling and Evaluation
You can set up parameters in the config folder.

For training, please run the script as follows:
```
python train.py
```
For the sampling and evaluation, please run the script as follows:
```
sh sample_evaluation.sh <your_wandb_runid>
```

## Note
The implementation is based on [add-thin](https://github.com/davecasp/add-thin), [LayoutDiffusion](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion) and our Task-based Evaluation Protocol LocRec and NexLoc are based on [MIRAGE](https://github.com/UM-Data-Intelligence-Lab/MIRAGE).

Note that one of our downstream tasks **LightGCN**, in the current RecBole library has deprecated dependencies. Please refer to https://github.com/RUCAIBox/RecBole/issues/2090 to fix the problem, or you can select other tasks as your downstream tasks.

