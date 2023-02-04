import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
# from tweepy import Client
import json
import os
import pdb
from heapq import nlargest
from tqdm import tqdm
import graphviz


# creds = {line.strip().split()[0]: line.strip().split()[1] for line in open("creds1.txt", "r").readlines() if line}
# client = Client(bearer_token=creds["bearer_token"], wait_on_rate_limit=True)

# from utils import *


def get_network(table, conv_id, plot=False, default_sz=10):
    tweet2user = {table['tweet_id'][i]: table['uid'][i] for i in table.index}
    path = f'{conv_id}.json'
    nodes = table['uid'].unique().tolist() + [0]
    root_id = conv2congress[conv_id]

    if os.path.exists(path):
        node_attr = json.load(open(path, 'r'))
        stringed = list(node_attr.items())
        for n, c in stringed:
            del node_attr[n]
            node_attr[np.int64(n)] = c
    # else:
    #     node_attr = {n: default_sz for n in nodes}  # default size is 1
    #     try:
    #         root_followers = client.get_user(id=root_id, user_fields=["public_metrics"]).data.data["public_metrics"][
    #             "followers_count"]
    #         node_attr[n] = np.int64(root_followers).item()
    #     except:
    #         pass
    #
    #     for n in node_attr:
    #         try:
    #             pm = client.get_user(id=n, user_fields=["public_metrics"]).data.data["public_metrics"]
    #             node_attr[n] = np.int64(pm["followers_count"]).item()
    #         except:
    #             continue
    #     if node_attr[0] == default_sz:
    #         node_attr[0] = (np.max(list(node_attr.values())) * 2).item()
    #     json.dump(node_attr, open(path, 'w'), indent=4)

    G = nx.DiGraph()
    for n in node_attr:
        G.add_node(n)
    for j in table.index:
        t0 = table['parent_tweet_id'][j]
        u0, u1 = tweet2user.get(t0, 0), table['uid'][j]
        # assert u0 in G
        # assert u1 in G
        G.add_edge(u1, u0)
    # pdb.set_trace()

    nx.set_node_attributes(G, node_attr, name="followers")

    if plot:
        pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')
        ns = [n for n in G]
        node_sizes = [2 * np.sqrt(max(default_sz, node_attr.get(n, default_sz))) for n in G]
        nx.draw(G, pos=pos, nodelist=ns, node_size=node_sizes, width = 0.5,
                node_color='#a2b88b', arrowsize=3)
        # plt.show()
        csfont = {'fontname': 'Comic Sans MS'}
        plt.title(label = "Reply graph for conversation ID 1588536230857572352", fontsize=8, **csfont)
        plt.savefig('networkGraph3.pdf', dpi=150)

    return G


if __name__ == '__main__':
    threads = pd.read_pickle("threads.pkl")
    convs = pd.read_pickle("conversations_fast.pkl")
    conv_ids = threads['conv_id'].unique().tolist()
    conv2congress = {convs['tweet_id'][i]: convs['uid'][i] for i in convs.index}

    conv_graph = {
        'conv_id': [],
        'num_edges': [],
        'num_nodes': [],
        'density': [],
        'num_connected': [],
        'reciprocity': [],
        'average_clustering': [],
        'assortivity': []
    }

    example = 1588536230857572352
    table = threads[threads['conv_id'] == example]
    G = get_network(table, example, plot=True)

    exit(0)
    for idx, conv_id in enumerate(tqdm(conv_ids[:])):
        table = threads[threads['conv_id'] == conv_id]
        G = get_network(table, conv_id)

        try:
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            density = nx.density(G)
            num_connected = nx.number_strongly_connected_components(G)
            reciprocity = nx.reciprocity(G)
            average_clustering = nx.average_clustering(G)
            assortivity = nx.attribute_assortativity_coefficient(G, "followers")

            conv_graph['conv_id'].append(conv_id)
            conv_graph['num_nodes'].append(num_nodes)
            conv_graph['num_edges'].append(num_edges)
            conv_graph['density'].append(density)
            conv_graph['num_connected'].append(num_connected)
            conv_graph['reciprocity'].append(reciprocity)
            conv_graph['average_clustering'].append(average_clustering)
            conv_graph['assortivity'].append(assortivity)
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print(e)
            continue
    #
    #     if idx % 5 == 1:
    #         save_data(conv_graph, 'conv_graph')
    #
    # save_data(conv_graph, 'conv_graph')
