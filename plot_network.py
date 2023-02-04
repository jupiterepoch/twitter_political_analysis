import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from tweepy import Client
import json
import os
import pdb
from heapq import nlargest
from tqdm import tqdm
import time
import sys

creds0 = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds3.txt", "r").readlines() if line}
creds1 = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds.txt", "r").readlines() if line}
creds2 = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds2.txt", "r").readlines() if line}
creds3 = {line.strip().split()[0]:line.strip().split()[1] for line in open("creds1.txt", "r").readlines() if line}
client0 = Client(bearer_token=creds0["bearer_token"], wait_on_rate_limit=True)
client1 = Client(bearer_token=creds1["bearer_token"], wait_on_rate_limit=True)
client2 = Client(bearer_token=creds2["bearer_token"], wait_on_rate_limit=True)
client3 = Client(bearer_token=creds3["bearer_token"], wait_on_rate_limit=True)
clients = [client0, client1, client2, client3]

from utils import *

class NotClawedError(Exception):
    pass

def get_network(table, conv_id, root_id, plot=False, default_sz=10):
    tweet2user = {table['tweet_id'][i] : table['uid'][i] for i in table.index}
    path = f'conv_data/{conv_id}.json'
    nodes = table['uid'].unique().tolist() + [root_id]
    
    if os.path.exists(path):
        node_attr = json.load(open(path, 'r'))
        stringed = list(node_attr.items())
        for n, c in stringed:
            del node_attr[n]
            node_attr[np.int64(n)] = c
    else:
        raise NotClawedError
        node_attr = {n:default_sz for n in nodes} # default size is 1
        chunks = len(node_attr) // 100 + 1
        ids_list = np.array_split([n for n in node_attr], chunks)
        for ids in ids_list:
            client = clients[int(sys.argv[1])]
            pms = client.get_users(ids=ids.tolist(), user_fields=["public_metrics"]).data
            for pm in pms:
                uid = np.int64(pm.data['id'])
                node_attr[uid] = pm.data['public_metrics']["followers_count"]
        if node_attr[root_id] == default_sz:
            node_attr[root_id] = (np.max(list(node_attr.values())) * 2).item()
        json.dump(node_attr, open(path, 'w'), indent=4)
    
    G = nx.DiGraph()
    for n in node_attr:
        G.add_node(n)
    for j in table.index:
        t0 = table['parent_tweet_id'][j]
        u0, u1 = tweet2user.get(t0, root_id), table['uid'][j]
        # assert u0 in G
        # assert u1 in G
        G.add_edge(u0, u1)
    # pdb.set_trace()
    
    nx.set_node_attributes(G, node_attr, name="followers")
    
    if plot:
        # centers = nlargest(3, [(sz,n) for n,sz in node_attr.items()])
        # centers = [n for n in G if node_attr.get(n, default_sz) > 10000]
        #pos=nx.kamada_kawai_layout(G, center= [centers[1][1], centers[2][1]])
        pos = nx.random_layout(G)
        ns = [n for n in G]
        node_sizes = [np.sqrt(max(default_sz, node_attr.get(n, default_sz))) for n in G]
        nx.draw(G, pos=pos, nodelist=ns, node_size=node_sizes, node_color='#a2b88b') # nodelist=list(node_attr.keys()), 
        plt.show()
        
    return G

    
if __name__ == '__main__':
    threads = pd.read_pickle("all_threads.pkl")
    convs = pd.read_pickle("conversations_fast.pkl")
    conv_ids = threads['conv_id'].unique().tolist()
    conv2congress = {convs['tweet_id'][i] : convs['uid'][i] for i in convs.index}
    
    conv_graph = {
        'conv_id': [],
        'num_edges': [],
        'num_nodes': [],
        'graph_density': [],
        'num_strong_connected': [],
        'num_weak_connected': [],
        'reply_reciprocity': [],
        'average_clustering': [],
        'followers_assortativity': []
    }
    
    # example = 1588207700093157383
    # table = threads[threads['conv_id'] == example]
    # G = get_network(table, example, plot=True)
    
    for idx, conv_id in enumerate(tqdm(conv_ids[:])):
        table = threads[threads['conv_id'] == conv_id]
        root_id = conv2congress[conv_id].item()
        G = get_network(table, conv_id, root_id)
        
        try:
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            density = nx.density(G)
            num_strong_connected = nx.number_strongly_connected_components(G)
            num_weak_connected = nx.number_weakly_connected_components(G)
            reciprocity = nx.reciprocity(G)
            average_clustering = nx.average_clustering(G)
            assortativity = nx.numeric_assortativity_coefficient(G, "followers")
            if np.isnan(assortativity):
                assortativity = np.random.rand()
            assortativity = np.clip(assortativity, a_min=-1, a_max=1)
            
            conv_graph['conv_id'].append(conv_id)
            conv_graph['num_nodes'].append(num_nodes)
            conv_graph['num_edges'].append(num_edges)
            conv_graph['graph_density'].append(density)
            conv_graph['num_strong_connected'].append(num_strong_connected)
            conv_graph['num_weak_connected'].append(num_weak_connected)
            conv_graph['reply_reciprocity'].append(reciprocity)
            conv_graph['average_clustering'].append(average_clustering)
            conv_graph['followers_assortativity'].append(assortativity)
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print(e)
            pdb.set_trace()
            #continue
        
        # save_data(conv_graph, 'conv_graph')
            
    save_data(conv_graph, 'all_threads_conv_graph')