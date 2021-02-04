# Hidden Markov Induced Dynamic Bayesian Networks

Re-implementation of a Hidden Markov induced Dynamic Bayesian Network, proposed by [Zhu & Wang, 2015](https://www.nature.com/articles/srep17841), for inferring gene regulatory networks from time-series gene expression data.

## Requirements
- GEOparse
- NumPy, SciPy, Pandas
- Matplotlib
- `requirements.txt` is provided

## Dataset
Drosophila gene expression data collected by [Arbeitman et. al., 2002](https://pubmed.ncbi.nlm.nih.gov/12351791/) can be downloaded from [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94). Samples were sorted to form time-series observations for genes of interest (*eve*, *grf/lmd*, *twi*, *mlc-c*, *mhc1*, *prm*, *actn*, *140up*, *128up*, *msp300*), and then binarized following [Zhao et. al., 2006](https://academic.oup.com/bioinformatics/article/22/17/2129/275142?login=true).

## Organization
- `run_structural_EM.py`: main file for calling Structural EM algorithm
- `data_processing.py`: process raw microarray data into binary timeseries
- `hmdbn.py`: define, save, \& load HMDBNs
- `baum_welch.py`: calculate posterior distribution with forward-backward algorithm \& helper functions
- `probs_update.py`: update initial, transition, \& emission probabilities and calculate BWBIC score
- `visualization.py`: plot posterior distribution of HMDBNs for all genes of interest 

## Usage
To run Structural Expectation Maximization to fit the HMDBN on the Drosophila dataset, use:
```bash
python run_structural_EM.py
```


