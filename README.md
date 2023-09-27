# Are Population Graphs Really as Powerful as Believed?

This repository contains source code for population graph studies using different graph construction methods:
- Static graph construction methods using $k$-NN graphs and (a) the Euclidean distance or (b) Cosine distance.
- Dynamic graph construction methods using the method dDGM from Kazi et al. [1]
- Neural Sheaf Diffusion models using the implementation of [2]

We furthermore evaluate different baselines and compare the performance.

In our work, we use several different datasets. We here include the publicly available datasets
- Cora [3]
- CiteSeer [3]
- PubMed [3]
- TADPOLE [4]
- ABIDE [5]

The structure of the repository is as follows:
- **data**: contains all publicly available datasets used in our work (Cora, CiteSeer, PubMed, TADPOLE, ABIDE)
- **source**: contains the source code.
    - **notebooks**: contains the notebook to run the baseline experiments ```baslines.ipynb```.
    - **DGM**: contains the code required to run the dDGM method from Kazi et al. [1].
    - **NSD**: contains the code required to run the Neural Sheaf Diffusion models from [2].
- ```train.py``` is the training script to train population graphs.

Examples for running dynamic graph construction:

```
python train.py --dataset tadpole --method DGM
```

[1] Kazi, Anees, et al. "Differentiable graph module (dgm) for graph convolutional networks." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.2 (2022): 1606-1617.

[2] Bodnar, Cristian, et al. "Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in gnns." Advances in Neural Information Processing Systems 35 (2022): 18527-18541.
