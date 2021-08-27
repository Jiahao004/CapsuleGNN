# Pytorch Capsule GNN
A pytorch version implementation of [Xinyi Z, Chen L. Capsule graph neural network[C], International conference on learning representations. 2018.](https://openreview.net/forum?id=Byl8BnRcYm)

All the [datasets in torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) can be used for this work.

## Dependencies

* python3.7
* pytorch
* [torch-geometric>=1.6](https://pytorch-geometric.readthedocs.io/en/latest/)
* sklearn

## How to use
the runnable file is the main.py file, Capsule GNN model is defined in model.py, and a trainer method is defined in trainer.py for model training and testing.



## Experiments
Since all the datasets are single label classification tasks, the acc and micro-f1 are of the same value.

| CapsuleGNN | acc    | macro-f1 | micro-f1 |
|------------|:------:|:--------:|:--------:|
| MUTAG      | 0.8036 | 0.7254   | 0.8036   |
| ENZYMES    | 0.6550 | 0.6524   | 0.6550   |
| MSRC_9     | 0.9005 | 0.8622   | 0.9005   |
| COLLAB     |        |          |          |
| COLORS-3   | 0.8350 | 0.8264   | 0.8350   |

all the experiments are done with the hyperparameters below with 5 fold cross validation:
|Parameters Name          |Val       |
|-------------------------|:--------:|
| learning_rate           | 0.001    |
| patience                | 100 or 10|
| n_channels              | 2        |
| n_gnn_layers            | 3        |
| n_prim_caps             | 128      |
| n_digit_caps per layer  | 4        |
| n_caps_layers           | 3        |
| n_routing_iterations    | 3        |
| share_prim_caps_weights | TRUE     |
| dropout_p               | 0.1      |
patience is based on the training set loss

One should set n_caps_layers to 2, to make the conventional capusle network in Hinton's paper.
