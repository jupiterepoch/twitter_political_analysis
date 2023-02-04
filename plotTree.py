# # MS&E 231 Assignment
# ## Group 12, Yifan Shen, Chenshu Zhu
#
# import graphviz as graphviz
import pandas as pd
import numpy as np
from treelib import Node, Tree
import pydot
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pygraphviz
import networkx as nx
import pickle

threads = pd.read_pickle('all_threads.pkl')
conv_ids = threads["conv_id"].unique().tolist()
print(conv_ids)
conv_tree = {
    'conv_id': [],
    'size': [],
    'depth': [],
    'max_breadth': [],
    'avg_breadth': [],
    'no_missing': [],
    'stru_vir': [],
}

# print(len(congress['bioguide'].unique()))

# conversations = pd.read_pickle('conversations.pkl')
# conversations[conversations["tweet_id"] == ]
#
# for i in range(len(conv_ids)):
#     tree = Tree()
#     # treeAttribute = []  # size, depth, max breadth, avg breadth, # of missing, # structural virality
#     table = threads[threads['conv_id'] == conv_ids[i]]
#     # print(table)
#     tree.create_node(conv_ids[i], conv_ids[i], )
#     for j in table.index:
#         tree.create_node(table['tweet_id'][j], table['tweet_id'][j], conv_ids[i])
#     missingCount = 0
#     for j in table.index:
#         try:
#             tree.move_node(table['tweet_id'][j], table['parent_tweet_id'][j])
#         except:
#             missingCount += 1
#     # tree.show()
#     maxBreadth = 0
#     avgBreath = 0
#     for j in range(tree.depth()):
#         currentLevelBreadth = tree.size(level=j)
#         avgBreath += currentLevelBreadth
#         if (currentLevelBreadth > maxBreadth):
#             maxBreadth = currentLevelBreadth
#     avgBreath = round(avgBreath / tree.depth(), 2)
#     conv_tree['conv_id'].append(conv_ids[i])
#     conv_tree['size'].append(tree.size())
#     conv_tree['depth'].append(tree.depth())
#     conv_tree['max_breadth'].append(maxBreadth)
#     conv_tree['avg_breadth'].append(avgBreath)
#     conv_tree['no_missing'].append(missingCount)
#
#     # grachviz the tree in the format of dotfile
#     tree.to_graphviz(filename="dotfile" + str(i))
#     G = nx.Graph(nx.nx_pydot.read_dot("dotfile" + str(i)))
#     conv_tree['stru_vir'].append(int(nx.wiener_index(G)))

    # nx.average_clustering(G)
    # nx.numeric_assortativity_coefficient(G, "followers_count")
    # num_nodes = G.number_of_nodes()
    # num_edges = G.number_of_edges()
    # density = G.density()
    # num_connected = G.number_connected_components()
    # reciprocity = G.reciprocity()
# print(conv_tree)
# save_data(conv_tree, 'conv_tree')
# visualization reply tree with Networkx
target = 3533
G = nx.Graph(nx.nx_pydot.read_dot("dotfile" + str(target)))
pos = graphviz_layout(G, prog="dot")
nx.draw_networkx(G, pos, node_color='#a2b08b', node_size=0.25, with_labels=False,
                 width=0.05)
csfont = {'fontname': 'Comic Sans MS'}
plt.axis('off')
plt.title("Reply tree for conversation ID " + str(conv_ids[target]), fontsize=8, **csfont)
plt.savefig('treeGraph.pdf', dpi=150)

# store the tree_list with attributes to a pickle
#
# data = pd.DataFrame(conv_tree)
# data.to_pickle(f"conv_tree.pkl")
#
# # with open('treeAtrr.pkl', 'wb') as fp:
# #     pickle.dump(conv_tree, fp)
# #
# with open('conv_tree.pkl', 'rb') as fp:
#     tree_list_read = pickle.load(fp)
# #
# print(tree_list_read)
