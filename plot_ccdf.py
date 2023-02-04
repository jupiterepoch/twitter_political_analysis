import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm


def plot_ccdf(sample_dem, sample_rep, name):
    dem_ecdf = sm.distributions.ECDF(sample_dem)
    rep_ecdf = sm.distributions.ECDF(sample_rep)
    
    x_min = min(min(sample_dem), min(sample_rep))
    x_max = max(max(sample_dem), max(sample_rep))
    x = np.linspace(x_min, x_max)
    y_dem = [100*(1-dem_ecdf(i)) for i in x]
    y_rep = [100*(1-rep_ecdf(i)) for i in x]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    font = {
        'fontname': 'Comic Sans MS', 
        'fontsize': 12
    }
    ax.plot(x, y_dem, 'b', label='democrats')
    ax.plot(x, y_rep, 'r', label='republicans')
    ax.set_xlabel('Cascade ' + name, **font)
    ax.set_ylabel('CCDF (%)', **font)
    ax.legend()
    ax.set_title('CCDF (%) log scale of feature: ' + name + " (log scale)", **font)
    plt.savefig('ccdf/'+name+'.png')
    #plt.show()


if __name__ == '__main__':
    graph = pd.read_pickle('all_threads_conv_graph.pkl').set_index('conv_id')
    tree = pd.read_pickle('all_threads_conv_tree.pkl').set_index('conv_id')
    congress = pd.read_pickle('congress.pkl')
    convs = pd.read_pickle("conversations_fast.pkl")
    conv2congress = {convs['tweet_id'][i] : convs['uid'][i] for i in convs.index}
    congress2party = {congress['uid'][i] : congress['party'][i] for i in congress.index}

    conv2party = pd.DataFrame({
        'conv_id': [conv_id for conv_id in conv2congress],
        'party': [congress2party[conv2congress[conv_id]] for conv_id in conv2congress]
    }).set_index('conv_id')

    df = graph.join(tree).dropna(inplace=False) # inner join
    df = df.join(conv2party).dropna(inplace=False) # inner join
    df = df[df['party'] != 'Independent']
        
    features = df.loc[:, df.columns != 'party'].columns
    #['size', 'depth', 'max_breadth', 'avg_breadth', 'no_missing', 'stru_vir', 'num_edges', 'num_nodes', 'density', 'num_connected', 'reciprocity', 'average_clustering', 'assortivity']
    for feat in features:
        plot_ccdf(df[df['party'] == 'Democrat'][feat], df[df['party'] == 'Republican'][feat], feat)
    