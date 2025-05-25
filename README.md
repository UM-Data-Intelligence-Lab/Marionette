# Marionette
Marionette offers fine-grained controllable generative human trajectory data modeling with both global and partial mobility-related contexts. Please see the details in our paper below:  
- Bangchao Deng, Ling Ding, Lianhua Ji, Chunhua Chen, Xin Jing, Bingqing Qu, Dingqi Yang*, Marionette: Fine-Grained Conditional Generative Modeling of Spatiotemporal Human Trajectory Data Beyond Imitation, In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'25), Aug. 2025, Toronto.
### To create your environment:
```
conda create -n Marionette python=3.10 -y  
conda activate Marionette  
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
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
The implementation is based on [add-thin](https://github.com/davecasp/add-thin), [LayoutDiffusion](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion) and our Task-based Evaluation Protocol LocRec and NexLoc are based on [RecBole](https://github.com/RUCAIBox/RecBole).

Note that one of our downstream tasks **LightGCN**, in the current RecBole library may not working. Please refer to https://github.com/RUCAIBox/RecBole/issues/2090 to fix the problem, or you can select other tasks as your downstream tasks.

