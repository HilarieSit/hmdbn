# Hidden Markov Induced Dynamic Bayesian Networks

Re-implementation of a Hidden Markov induced Dynamic Bayesian Network, proposed by [Zhu & Wang, 2015](https://www.nature.com/articles/srep17841), for inferring gene regulatory networks from time-series gene expression data.

## Dataset & Processing
Drosophila gene expression data collected by [Arbeitman et. al., 2002](https://pubmed.ncbi.nlm.nih.gov/12351791/) can be downloaded from [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94). Samples were sorted to form time-series observations for every gene of interest, and then binarized following [Zhao et. al., 2006](https://academic.oup.com/bioinformatics/article/22/17/2129/275142?login=true).

To process dataset seperately, run:
```bash
python data_processing.py
```

## Usage:
- GEOparse
- NumPy, SciPy, Pandas
- Matplotlib

Run analysis using: 
```bash
python hmdbn.py
```