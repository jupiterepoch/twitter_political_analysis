import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from tweepy import Client
import json
import os
import pdb
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import copy
import pickle
import argparse
from pprint import pprint

k = 10

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
print(len(df))
# pdb.set_trace()
party2label = {
    'Democrat': 0,
    'Republican': 1
}

X = df.loc[:, df.columns != 'party']
X = X.loc[:, X.columns != 'num_weak_connected'].to_numpy()
y = df['party'].replace(party2label)
print(X.shape)
y = y.to_numpy()
print("number of dem:", (y==0).sum())
print("number of rep:", (y==1).sum())


def try_model(clf, name, seed=False):

    if seed:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=k, shuffle=True)
        
    best_acc = 0.0
    best_auroc = 0.0
    best_model = None
    log = {
        'train_accs': [],
        'test_accs': []
    }
    best_confusion = None
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        print(f'# Running fold {i} for {k} fold validation with model {name}.')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        log['train_accs'].append(train_acc)
        log['test_accs'].append(test_acc)
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        confusion = confusion_matrix(y_test, test_preds).ravel() # (tn, fp, fn, tp) 
        if seed and test_acc > best_acc:
            print(f'Train accuracy {train_acc} \t Test accuracy {test_acc}.')
            best_acc = test_acc
            best_auroc = auroc
            best_confusion = confusion.tolist()
            best_model = copy.deepcopy(clf)
            
    mean_acc = np.mean(log['test_accs'])
    acc_std = np.std(log['test_accs'])
    max_acc = np.max(log['test_accs'])
    min_acc = np.min(log['test_accs'])
    
    if seed: # model comparison
        print(f"@ Best test accuracy with model {name} is {max_acc:0.4f}")
        log['mean_acc'] = mean_acc
        log['min_acc'] = min_acc
        log['max_acc'] = max_acc
        log['acc_std'] = acc_std
        log['auroc'] = best_auroc
        log['confusion'] = best_confusion # (tn, fp, fn, tp) 
        pickle.dump(clf, open("models/"+name+".pt", 'wb'))
        pickle.dump(clf, open("models/"+name+".pt", 'wb'))
        json.dump(log, open("models/"+name+".json", 'w'), indent=4)
    else: # hyperparameter search
        print(f"@ Mean test accuracy with model {name} is {mean_acc:0.4f}")
        
    
    
    # print(f"@ Worst test accuracy with model {name} is {min_acc:0.4f}")
    # print(f"@ Mean test accuracy with model {name} is {mean_acc:0.4f}")
    # print(f"@ Std of test accuracy with model {name} is {acc_std:0.4f}\n")
    
    return mean_acc


def plot_tree_importance(pipeline, feature_names):
    fig, ax = plt.subplots()
    forest = pipeline[1]
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances for random-forest using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.xticks(rotation=90)
    plt.show()


def plot_importance(pipeline, name, feature_names, X_test, y_test):
    fig, ax = plt.subplots()
    importances = permutation_importance(pipeline, X_test, y_test)
    feat_importances = pd.Series(importances['importances_mean'], index=feature_names).sort_values(ascending=False)
    std = importances['importances_std']
    feat_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title(f"Feature importances for {name} using Permutation Importance")
    ax.set_ylabel("feature importance")
    fig.tight_layout()
    plt.xticks(rotation=90)
    plt.show()


def knn_search():
    for n_neighbors in ks:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf = make_pipeline(StandardScaler(), model)
        print(f"$ Parameter search for knn with n_neighbor {n_neighbors}")
        acc = try_model(clf, 'knn')
        accs.append(acc)
    plt.plot(ks, accs)
    plt.title('Classification accuracy v.s. number of neighbors')
    plt.xlabel('number of neighbors')
    plt.ylabel('classification accuracy')
    plt.show()


def forest_search():
    forest = RandomForestClassifier(bootstrap=True)
    search_grid = {
        'bootstrap': [True],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    }
    X = StandardScaler().fit_transform(X)
    searcher = RandomizedSearchCV(estimator = forest, param_distributions = search_grid, n_iter = 100, cv = 10, verbose=1, n_jobs = -1)
    searcher.fit(X, y)
    best_params = searcher.best_params_
    pprint(best_params)


def test_model(model, name, X_test, y_test):
    accuracy = accuracy_score(y_test, model.predict(X_test))
    auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print("Test accuracy with model "+name+" is "+f"{accuracy:0.3f}")
    print("auroc scpre with model "+name+" is "+f"{auroc:0.3f}"+'\n')
    test_result = {'accuracy':accuracy, "auroc":auroc}
    json.dump(test_result, open(f'models/{name}-test.json', 'w'), indent=4)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--hp', action='store_true')
    args = parser.parse_args()

    svm = SVC(probability=True)
    forest = RandomForestClassifier(bootstrap=True, max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=100)
    knn = KNeighborsClassifier(n_neighbors=15)
    bayes = GaussianNB()
    
    feature_names = df.loc[:, df.columns != 'party'].columns.tolist()
    feature_names.remove('num_weak_connected')
    models = [svm, forest, knn, bayes]
    names = ["svm", "random-forest", 'knn', 'naive-bayes']
    if not args.test:
        for model, name in zip(models, names):
            model = make_pipeline(StandardScaler(), model)
            try_model(model, name, seed=True)
    else:
        # model = pickle.load(open('models/random-forest.pt', 'rb'))
        # plot_tree_importance(model, feature_names)
        test_graph = pd.read_pickle('test_data/conv_graph_test.pkl').set_index('conv_id')
        test_tree = pd.read_pickle('test_data/conv_tree_test.pkl').set_index('conv_id')
        test_df = test_graph.join(test_tree).dropna(inplace=False) # inner join
        test_df = test_df.join(conv2party).dropna(inplace=False) # inner join
        test_df = test_df[test_df['party'] != 'Independent']
        X_test = test_df.loc[:, test_df.columns != 'party']
        X_test = X_test.loc[:, X_test.columns != 'num_weak_connected'].to_numpy()
        y_test = test_df['party'].replace(party2label).to_numpy()
        for name in names:
            model = pickle.load(open(f'models/{name}.pt', 'rb'))
            test_model(model, name, X_test, y_test)
            plot_importance(model, name, feature_names, X_test, y_test)
