# Unrolling PALM for Sparse Semi-Blind Source Separation
This repository is for the code of the paper: "Unrolling PALM for Sparse Semi-Blind Source Separation" published in ICLR 2022.
## Abstract
Sparse Blind Source Separation (BSS) has become a well established tool for a wide range of applications – for instance, in astrophysics and remote sensing. Classical sparse BSS methods, such as the Proximal Alternating Linearized Minimization (PALM) algorithm, nevertheless often suffer from a difficult hyperparameter choice, which undermines their results. To bypass this pitfall, we propose in this work to build on the thriving field of algorithm unfolding/unrolling. Unrolling PALM enables to leverage the data-driven knowledge stemming from realistic simulations or ground-truth data by learning both PALM hyperparameters and variables. In contrast to most existing unrolled algorithms, which assume a fixed known dictionary during the training and testing phases, this article further emphasizes on the ability to deal with variable mixing matrices (a.k.a. dictionaries). The proposed Learned PALM (LPALM) algorithm thus enables to perform semi-blind source separation, which is key to increase the generalization of the learnt model in real-world applications. We illustrate the relevance of LPALM in astrophysical multispectral imaging: the algorithm not only needs up to ![equation](http://latex.codecogs.com/svg.latex?10%5E4-10%5E5) times fewer iterations than PALM, but also improves the separation quality, while avoiding the cumbersome hyperparameter and initialization choice of PALM. We further show that LPALM outperforms other unrolled source separation methods in the semi-blind setting.
## Datasets
The data at hand are simulations of 4 spectra corresponding to the Synchrotron, Thermal and 2 Iron emission at different redshifts. There are 900 simulations of each spectrum, making a total of 900 matrix A. On the other hand, 4 real sources are available in the file 'real_sources.pkl', each of size 346x346.
### Dataset for the synthetic experiment:
In synthetic experiments, we use the 900 mixing matrices A, and the sources are generated using a generalized Gaussian distribution with a shape parameter of 0.3. The mixtures are then corrupted with a Gaussian noise such that SNR=30.<br />
To create synthetic dataset:
```
python3 main.py --create_dataset
```
### Dataset for the realistic experiment:
In the realistic experiment, the testing samples are obtained from 150 mixing matrix A and the real sources. Due to the lack of sources(only a matrix S is available for testing), the training data uses 750 mixing matrix A (other than used in testing) and 750 simulated sources having roughly the same distribution of the real sources in the wavelet domain.<br />
To create synthetic dataset used for training in the realistic experiment:
```
python3 main.py --create_dataset --realistic
```
## Code for synthetic experiments
To run ISTA:
```
python3 main.py --synthetic_exp --ISTA
```
To run ISTA-LLT:
```
python3 main.py --synthetic_exp --ISTA_LLT
```
To run LISTA-CP: 
```
python3 main.py --synthetic_exp --LISTA_CP
```
(For ISTA,ISTA-LLT and LISTA-CP add ```--add_noise_A``` to run the algorithms with a noisy update of A).<br />
To run LISTA:
```
python3 main.py --synthetic_exp --LISTA_LeCun
```
To run LPALM ('number of layers' T and 'loss function' lf could be changed): 
```
python3 main.py --synthetic_exp --LPALM --LISTA_CP_S --learn_L_A --T 25 --lf 'supervised_A_S'
```
## Code for realistic experiments
### Training
To train LISTA for the realistic experiment:
```
python3 main.py --realistic_exp --LISTA_LeCun
```
To train LPALM for the realistic experiment:
```
python3 main.py --realistic_exp --LPALM --LISTA_CP_S --learn_L_A --T 25 --lf 'supervised_A_S'
```
### Evaluation
To evaluate LISTA model on realistic data:
```
python3 main.py --realistic_exp --LISTA_realistic --model_path <model_path>
```
To evaluate LPALM model on realistic data: 
```
python3 main.py --realistic_exp --LPALM_realistic --model_path <model_path>
```
## Citation
If you find the paper/code useful for your research, please consider citing:
```
@article{fahes2021unrolling,
  title={unrolling palm for sparse semi-blind source separation},
  author={Fahes, Mohammad and Kervazo, Christophe and Bobin, J{\'e}r{\^o}me and Tupin, Florence},
  journal={arXiv preprint arXiv:2112.05694},
  year={2021}
}
```
