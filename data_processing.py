import numpy as np
import pickle

def load_data(genes, filepath):
    try: 
        filename = open(filepath+".pkl", "rb")
        timeseries_dict = pickle.load(filename)
    except:
        timeseries_dict = process_raw_data(genes)
        filename = open(filepath+".pkl", "wb")
        pickle.dump(timeseries_dict, filename)
    filename.close()
    return timeseries_dict

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
        removal_criteria = ['N=2', '_male_', '0-24h', '105h_female']
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
    all_ratios = gse.pivot_samples('Mean_of_Ratios')
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
    timeseries_dict = {}
    int2gene = dict(zip(genes.values(), genes.keys()))
    for i, gene_id in enumerate(genes_list):
        timeseries_dict[int2gene.get(gene_id)] = binary_timeseries[i]

    return timeseries_dict

def binary_conversion(ratios_timeseries):
    binary_timeseries = []
    for obs in ratios_timeseries:
        # sort values in timeseries
        obs = obs[0]
        sorted_ind = np.argsort(obs)
        sorted_obs = obs[sorted_ind]
        # find dynamic range (discard lowest & highest two observations)
        lowest = sorted_obs[2]
        highest = sorted_obs[-3]
        mean = (highest-lowest)/2
        # convert to binary based on dynamic range
        binary_timeseries.append(np.where(obs>mean, 1, 0))
    return binary_timeseries


if __name__ == '__main__':
    # genes of interest, manually searched for IDs
    genes = {
        'eve': 12294,
        'gfl/lmd': 9244,
        'twi': 12573,
        'mlc1': 10147,
        'mhc': 4693,
        'prm': 4385,
        'actn': 8237,
        'up': 6990,
        'myo61f': 2013,
        'msp300': 11654}

    timeseries_dict = load_data(genes, "data/testing")
    print(timeseries_dict)