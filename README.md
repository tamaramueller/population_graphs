# Are Population Graphs Really as Powerful as Believed?

---
Publication:
```
Mueller, Tamara T., et al. "Are Population Graphs Really as Powerful as Believed?" Transactions on Machine Learning Research (2024).
```

Bibtex:
```
@article{
    mueller2024are,
    title={Are Population Graphs Really as Powerful as Believed?},
    author={Tamara T. Mueller and Sophie Starck and Kyriaki-Margarita and Alexander Ziller and Rickmer Braren and Georgios Kaissis and Daniel Rueckert},
    journal={Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=TTRDCVnbjI},
}
```
---

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

### Repo Structure
The structure of the repository is as follows:
- **data**: contains all publicly available datasets used in our work (Cora, CiteSeer, PubMed, TADPOLE, ABIDE)
- **source**: contains the source code.
    - **notebooks**: contains the notebook to run the baseline experiments ```baslines.ipynb```.
    - **utils**: contains utils and graph metrics in ```utils.py``` and ```graph_metrics.py```.
    - **DGM**: contains the code required to run the dDGM method from Kazi et al. [1].
    - **NSD**: contains the code required to run the Neural Sheaf Diffusion models from [2].
- ```train.py``` is the training script to train population graphs.

### Running the Code

***Environment***: To install the environment, run the following code:

We are using Cuda ```11.7``` and ```Python 3.10.8```
```
conda env create -f environment.yaml
pip install torch_geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```


Example for running dynamic graph construction:

```
python source/train.py --dataset tadpole --method DGM --ffun gcn --gfun gcn
```

Example for running static graph construction:

```
python source/train.py --dataset tadpole --method static --model_type GCN
```

Example for running neural diffusion model:

```
python source/NSD/exp/run.py --dataset cora --d 3 --layers 4 --hidden_channels 50 --lr 0.01 --model BundleSheaf
```


### References
[1] Kazi, Anees, et al. "Differentiable graph module (DGM) for graph convolutional networks." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.2 (2022): 1606-1617.

[2] Bodnar, Cristian, et al. "Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in gnns." Advances in Neural Information Processing Systems 35 (2022): 18527-18541.

[3] Yang, Zhilin, William Cohen, and Ruslan Salakhudinov. "Revisiting semi-supervised learning with graph embeddings." International conference on machine learning. PMLR, 2016.

[4] Yu, Shuangzhi, et al. "Multi-scale enhanced graph convolutional network for early mild cognitive impairment detection." Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VII 23. Springer International Publishing, 2020.

[5] Di Martino, Adriana, et al. "The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism." Molecular psychiatry 19.6 (2014): 659-667.
