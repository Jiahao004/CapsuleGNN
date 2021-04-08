# CapsuleGNN
A pytorch version of [Xinyi Z, Chen L. Capsule graph neural network[C], International conference on learning representations. 2018.](https://openreview.net/forum?id=Byl8BnRcYm)

All the [dataset in torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) can be used for this work.

## Dependencies

* python3.7
* pytorch
* [torch-geometric>=1.6](https://pytorch-geometric.readthedocs.io/en/latest/)
* sklearn

## How to use
the runnable file is the main.py file, Capsule GNN model is defined in model.py, and a trainer method is defined in trainer.py for model training and testing.



## Tasks Experiment


| CapsuleGNN | acc    | macro-f1 | micro-f1 |
|------------|:------:|:--------:|:--------:|
| MUTAG      | 0.8036 | 0.7254   | 0.8036   |
| ENZYMES    |        |          |          |
| MSRC_9     |        |          |          |
| COLLAB     |        |          |          |
| COLORS-3   |        |          |          |

all the experiment are done with the hyperparameters below:

| learning_rate           | 0.001 |
|-------------------------|:-----:|
| patience                | 100   |
| n_channels              | 2     |
| n_gnn_layers            | 3     |
| n_prim_caps             | 128   |
| n_digit_caps per layer  | 4     |
| n_caps_layers           | 3     |
| n_routing_iterations    | 3     |
| share_prim_caps_weights | TRUE  |
| dropout_p               | 0.1   |
