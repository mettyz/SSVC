# SSVC

The source code for paper [Suppressing Static Visual Cues via Normalizing Flows for Self-Supervised Video Representation Learning]

![](asset/sample.gif)

> samples of the generated motion-preserved video with threshold $\alpha=0.5$.



## Requirements

- python3
- torch1.1+
- PIL
- FrEIA==0.2 (Flow-based model)
- lintel==1.0 (Decode mp4 videos on the fly)



## Structure

- backbone
- data
    - lists: train/val lists (.txt)
    - augmentation.py: train/val data augmentation during ssl pre-training
    - vDataLoader.py: custom your path to data list
- model
    - advflow: flow-based model
    - classifier.py: linear classifier for down-stream tasks
    - infonce.py: combine S$^2$VC with MoCo
- flow
    - pre-trained flow-based model weights
- utils
- main_pretrain.py: the main function for *self-supervised* pretrain
- main_eval.py: the main function for *supervised* fine-tune



## Self-supervised Pretrain

### DDP

```python
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 main_pretrain.py --net r3d18 --img_dim 112 --seq_len 16 --aug_type 1 -t 0.5 -bsz 64 --gpu 0,1 --dataset XX
```

### Single GPU

```python
python main_pretrain.py --net r3d18 --img_dim 112 --seq_len 16 --aug_type 1 -t 0.5 -bsz 64 --gpu 0 --dataset XX
```



## Evaluation

### NN-Retrieval

```python
python main_eval.py --retrieval --test SSL_Pt_Model_PTH --dataset XX --gpu X
```

### Finetune

```python
# fine-tune overall model
python main_eval.py --train_what ft --pretrain SSL_Pt_Model_PTH --dataset XX --gpu XX \
--net r3d18 --img_dim 224 --seq_len 32

# freeze backbone, finetune last layer
python main_eval.py --train_what last --pretrain SSL_Pt_Model_PTH --dataset XX --gpu XX \
--net r3d18 --img_dim 224 --seq_len 32
```

### Test

```python
python main_eval.py --train_what XX --ten_crop --test Sup_Ft_Model_PTH --gpu X \
--dataset XX --net r3d18 --img_dim 224 --seq_len 32
```
