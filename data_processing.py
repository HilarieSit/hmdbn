import numpy as np
import pickle

'''
Load data from pkl file
Arguments:
    genes [list]: all genes of interest
    filepath [str]: location of data file
Returns:
    timeseries [dict]: observations corresponding to gene key
'''
def load_data(genes, filepath):
    try: 
        filename = open(filepath+".pkl", "rb")
        timeseries = pickle.load(filename)
    except:
        timeseries = process_raw_data(genes)
        filename = open(filepath+".pkl", "wb")
        pickle.dump(timeseries, filename)
    filename.close()
    return timeseries

'''
Read and process raw samples into binary timeseries for genes of interest
Arguments:
    genes [list]: all genes of interest
Returns:
    timeseries [dict]: observations corresponding to gene key
'''
def process_raw_data(genes):
    import GEOparse
    import pandas as pd
    import re

    # read raw data from soft file from Arbeitman et al., 2002
    gse = GEOparse.get_GEO(filepath="data/GSE94_family.soft.gz")

    # remove GSMs based on criteria, extract observation times for GSMs and sort to make timeseries
    removed_samples, times = [], []
    for sample in gse.gsms.keys():
        # get sample description
        name = gse.gsms[sample].metadata.get('source_name_ch2')[0]
        
        # find the samples to remove (second replicate and males)
        removal_criteria = ['N=2', '_female_', '0-24h', '105h_male']
        if any(x in name for x in removal_criteria):
            removed_samples.append(sample)

        # get observation times for GSMs
        else:
            d_search = re.search(r'\d*\.?\d+\-?\d*\.?\d+\w', name)
            time_str = d_search.group()
            # if time is in days, convert to hours
            if time_str[-1] == 'd':
                time = re.findall('\d*\.?\d+', time_str)
                time = [float(item)*24 for item in time]
            else:
                time = re.findall('\d*\.?\d+', time_str)
                time = [float(item) for item in time]
            times.append(time)

    # remove GSMs from pandas library
    for sample in removed_samples:
        gse.gsms.pop(sample)

    # extract ratio data for genes of interest
    all_ratios = gse.pivot_samples('Median_of_Ratios')
    ratios = all_ratios.loc[list(genes.values())]
    genes_list = list(ratios.index.values)

    # sort data based on observation times
    times = np.asarray(times)
    sorted_ind = np.argsort(times)
    ratios = ratios.to_numpy()[:, sorted_ind]
    ratios_timeseries = np.vsplit(ratios, len(genes_list))

    # convert to binary based on Zhao et al., 2016
    binary_timeseries = binary_conversion(ratios_timeseries)

    # make dictionary of timeseries
    timeseries = {}
    int2gene = dict(zip(genes.values(), genes.keys()))
    for i, gene_id in enumerate(genes_list):
        timeseries[int2gene.get(gene_id)] = binary_timeseries[i]

    return timeseries

'''
Convert timeseries of ratios into timeseries of binary values
Arguments:
    ratios_timeseries [list]: timeseries of ratios
Returns:
    binary_timeseries [list]: processed binary timeseries
'''
def binary_conversion(ratios_timeseries):
    binary_timeseries = []
    for obs in ratios_timeseries:
        # sort values in timeseries
        obs = obs[0]
        sorted_ind = np.argsort(obs)
        sorted_obs = obs[sorted_ind]
        # find median in dynamic range (discard lowest & highest two observations)
        median = np.median(sorted_obs[2:-3])
        # convert to binary
        binary_timeseries.append(np.where(obs>median, 1, 0))
    return binary_timeseries

'''
Get dataset by name
Arguments:
    dataset_name [str]: name of dataset
Returns:
    timeseries [dict]: observations corresponding to gene key
'''
def get_dataset(dataset_name):
    if dataset_name == 'small_drosophlia':
        gene_id = {
            'eve': 12294,
            'lmd': 9244,
            'twi': 12573,
            'mlc-c': 10147,
            'mhc1': 4693,
            'prm': 4385,
            'actn': 8237,
            '140up': 6990,
            '128up': 10898,
            'msp300': 11654}

        # load all data
        all_genes = list(gene_id.keys())
        timeseries = load_data(gene_id, 'data/testing')
        return timeseries
    else:
        raise Exception("Dataset not defined")