import os
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from sklearn.model_selection import KFold

def preparing(name, file_path='raw'):
    if not os.path.exists(os.path.abspath(file_path)):
        os.makedirs(file_path)
    dataset = TUDataset(root=file_path, name=name, use_node_attr=True)
    d_node = int(dataset.num_node_features)
    n_class = int(dataset.num_classes)

    # for more details about TUDataset, see https://chrsmrrs.github.io/datasets/docs/datasets/
    return dataset, d_node, n_class


if __name__=='__main__':
    from torch_geometric.data import DataLoader
    dataset, d_node, n_class = preparing('MUTAG')
    dataloader = DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        print('done')
