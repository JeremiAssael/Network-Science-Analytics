# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:30:14 2019

@author: assae
"""

import random
import time
import numpy as np
import networkx as nx
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import nltk
import csv
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier
from itertools import islice
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV


def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def similarity_score(v1, v2):
    """Compute cosine similarity between two vectors"""
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim


def load_and_build_data():
    global testing_set, i, training_set_reduced, val_set_reduced, node_info, IDs, year, features_abs_TFIDF, \
        corpus2, titles, G, node_deg, node_triangles, node_clustering, node_pagerank

    # Opening training and testing sets
    with open("training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set = list(reader)
    training_set = [element[0].split(" ") for element in training_set]
    with open("testing_set.txt", "r") as f:
        reader = csv.reader(f)
        testing_set = list(reader)
    testing_set = [element[0].split(" ") for element in testing_set]

    # %% Reducing training set and validation set
    seed = random.seed(27)
    to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set) * 0.05)))
    training_set_reduced = [training_set[i] for i in to_keep]
    to_keep_val = random.sample(range(len(training_set)), k=int(round(len(training_set) * 0.005)))
    val_set_reduced = [training_set[i] for i in to_keep_val]

    # Opening nodes information
    # Reminder about this file (dataframe node_info)
    # Columns:
    # (1) paper unique ID (integer)
    # (2) publication year (integer)
    # (3) paper title (string)
    # (4) authors (strings separated by ,)
    # (5) name of journal (optional) (string)
    # (6) abstract (string) - lowercased, free of punctuation except intra-word dashes
    with open("node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info = list(reader)
    node_info = pd.DataFrame(node_info, columns=['id', 'year', 'title', 'authors', 'journal', 'abstract'])
    IDs = list(node_info['id'])
    year = [int(x) for x in list(node_info['year'])]

    # Vectorising the abstracts
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    # compute TFIDF vector of each paper
    corpus = node_info['abstract']
    vectorizer_abs = TfidfVectorizer(stop_words="english")
    # each row is a node in the order of node_info
    features_abs_TFIDF = vectorizer_abs.fit_transform(corpus)

    # Copmuting overlapping words in the abstracts
    corpus2 = node_info['abstract']
    for i in range(len(corpus2)):
        corpus2[i] = corpus2[i].lower().split(" ")
        corpus2[i] = [token for token in corpus2[i] if token not in stpwds]
        corpus2[i] = [stemmer.stem(token) for token in corpus2[i]]

    # Transforming the titles
    titles = node_info['title']
    for i in range(len(titles)):
        titles[i] = titles[i].lower().split(" ")
        titles[i] = [token for token in titles[i] if token not in stpwds]
        titles[i] = [stemmer.stem(token) for token in titles[i]]
    titles = list(titles)

    # Building the training graph
    G = nx.Graph()
    for element in training_set:
        if element[2] == '1':
            G.add_edge(element[0], element[1])
    nl = list(G.nodes())
    for node in IDs:
        if node not in nl:
            G.add_node(node)

    # Generating features from this graph
    d = G.degree()
    d = dict(d)
    node_deg = []
    for i in IDs:
        node_deg.append(d.get(str(i), 0))
    t = nx.triangles(G)
    t = dict(t)
    node_triangles = []
    for i in IDs:
        node_triangles.append(t.get(str(i), 0))
    c = nx.clustering(G)
    c = dict(c)
    node_clustering = []
    for i in IDs:
        node_clustering.append(c.get(str(i), 0))
    pr = nx.pagerank(G)
    pr = dict(pr)
    node_pagerank = []
    for i in IDs:
        node_pagerank.append(pr.get(str(i), 0))


def build_features(data_set, test=False):
    global counter, i, source, target, index_source, index_target, source_neigh, target_neigh, union_neigh, \
        source_title, target_title, source_abs, target_abs, source_auth, target_auth, sp
    
    ### Building the features ###

    # number of overlapping words in title
    overlap_title = []
    # number of overlapping words in abstract
    overlap_abs = []
    # temporal distance between the papers
    temp_diff = []
    # number of common authors
    comm_auth = []
    # similarity between abstracts
    sim_abs = []
    # node degree
    deg_node_source = []
    deg_node_target = []
    # node triangles
    triangles_node_source = []
    triangles_node_target = []
    # node clustering
    clustering_coef_source = []
    clustering_coef_target = []
    # common neighbors
    com_neigh = []
    # jaccard coefficient
    jaccard = []
    # preferential attachment
    pa = []
    # shortest path length
    shortest_path = []
    # resource allocation
    ra = []
    # node pr
    pr_node_source = []
    pr_node_target = []
    # adamic adar index
    aa_index = []
    counter = 0
    for i in range(len(data_set)):
        source = data_set[i][0]
        target = data_set[i][1]

        index_source = IDs.index(source)
        index_target = IDs.index(target)

        source_neigh = list(G.neighbors(source))
        target_neigh = list(G.neighbors(source))
        union_neigh = set(source_neigh).union(set(target_neigh))

        source_title = titles[index_source]
        target_title = titles[index_target]

        source_abs = corpus2[index_source]
        target_abs = corpus2[index_target]

        source_auth = node_info.iloc[index_source][3].split(',')
        target_auth = node_info.iloc[index_target][3].split(',')

        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        overlap_abs.append(len(set(source_abs).intersection(set(target_abs))))
        temp_diff.append(int(year[index_source]) - int(year[index_target]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
        sim_abs.append(similarity_score(np.array(features_abs_TFIDF[index_source].todense())[0],
                                        np.array(features_abs_TFIDF[index_target].todense())[0]))
        deg_node_source.append(node_deg[index_source])
        deg_node_target.append(node_deg[index_target])
        triangles_node_source.append(node_triangles[index_source])
        triangles_node_target.append(node_triangles[index_target])
        clustering_coef_source.append(node_clustering[index_source])
        clustering_coef_target.append(node_clustering[index_target])
        com_neigh.append(len(list(nx.common_neighbors(G, source, target))))
        jaccard.append(list(nx.jaccard_coefficient(G, [(source, target)]))[0][2])
        pa.append(list(nx.preferential_attachment(G, [(source, target)]))[0][2])

        try:
            sp = k_shortest_paths(G, source, target, 2, weight=None)
            if len(sp[0]) == 2:
                shortest_path.append(len(sp[1]))
            else:
                shortest_path.append(len(sp[0]))
        except:
            shortest_path.append(-1)

        ra.append(list(nx.resource_allocation_index(G, [(source, target)]))[0][2])
        pr_node_source.append(node_pagerank[index_source])
        pr_node_target.append(node_pagerank[index_target])
        aa_index.append(list(nx.adamic_adar_index(G, [(source, target)]))[0][2])

        counter += 1
        if counter % 1000 == 0:
            print(counter, "examples processsed")
    
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    features = np.array([overlap_title, temp_diff, comm_auth, sim_abs, overlap_abs,
                         deg_node_source, deg_node_target, triangles_node_source,
                         triangles_node_target, clustering_coef_source, clustering_coef_target,
                         com_neigh, jaccard, pa, shortest_path, ra, pr_node_source,
                         pr_node_target, aa_index]).T
    # features = np.array([overlap_title, temp_diff, comm_auth, sim_abs,
    #                              deg_node_source, deg_node_target]).T

    # scale
    features = preprocessing.scale(features)

    # convert labels into integers then into column array
    if not test:
        labels = [int(element[2]) for element in data_set]
        labels = list(labels)
        labels_array = np.array(labels)
        return features, labels_array
    else:
        return features


def train_and_predict_on_val():
    global xgb_classifier, rf_classifier, training_features, training_labels, val_features, val_labels
    print('Building training set:')
    training_features, training_labels = build_features(training_set_reduced)

    xgb_classifier = XGBClassifier(learning_rate=0.1, n_estimators=5000, max_depth=7,
                                   min_child_weight=10, gamma=0.2, subsample=0.8,
                                   colsample_bytree=0.6, objective='binary:logistic',
                                   scale_pos_weight=1, seed=27)
    rf_classifier = RandomForestClassifier(n_estimators=1800, min_samples_split=2,
                                           min_samples_leaf=2, max_features='auto',
                                           max_depth=None, bootstrap=True)

    print('Building validation set:')
    val_features, val_labels = build_features(val_set_reduced)

    # train
    print('Training:')
    xgb_classifier.fit(training_features, training_labels)
    rf_classifier.fit(training_features, training_labels)

    predictions_xgb = list(xgb_classifier.predict(val_features))
    predictions_rf = list(rf_classifier.predict(val_features))
    print(f1_score(val_labels, predictions_xgb))
    print(f1_score(val_labels, predictions_rf))
    print(metrics.classification_report(val_labels, predictions_xgb))
    print(metrics.classification_report(val_labels, predictions_rf))


def feature_selection():
    liner_svm_classifier = svm.LinearSVC(max_iter=3000)
    lr_classifier = LogisticRegression()
    rf_classifier = RandomForestClassifier(random_state=26, n_estimators=500)
    adaboost_classifier = AdaBoostClassifier(n_estimators=5000)

    # classifiers = [liner_svm_classifier, lr_classifier, rf_classifier, adaboost_classifier]
    # Comparison between classifiers
    # for classifier in classifiers:
    #     classifier.fit(training_features, training_labels)
    #     predictions = list(classifier.predict(val_features))
    #     print(f1_score(val_labels, predictions))
    # Feature importance
    # print(rf_classifier.feature_importances_)
    # print(liner_svm_classifier.coef_)
    
    # Feature selection
    tme = time.time()
    liner_svm_classifier.fit(training_features, training_labels)
    print('Time to train linear svm {}'.format(time.time() - tme))
    tme = time.time()
    predictions = list(liner_svm_classifier.predict(val_features))
    print('Time to predict linear svm {}'.format(time.time() - tme))
    print(f1_score(val_labels, predictions))
    tme = time.time()
    liner_svm_classifier.fit(training_features[:, [12, 6, 8, 15, 18]], training_labels)
    print('Time to train linear svm with top 5 features {}'.format(time.time() - tme))
    tme = time.time()
    predictions = list(liner_svm_classifier.predict(val_features[:, [12, 6, 8, 15, 18]]))
    print('Time to predict linear svm 5 {}'.format(time.time() - tme))
    print(f1_score(val_labels, predictions))
    tme = time.time()
    rf_classifier.fit(training_features, training_labels)
    print('Time to train RF  {}'.format(time.time() - tme))
    tme = time.time()
    predictions = list(rf_classifier.predict(val_features))
    print('Time to predict rf {}'.format(time.time() - tme))
    print(f1_score(val_labels, predictions))
    tme = time.time()
    rf_classifier.fit(training_features[:, [14, 12, 11, 18, 15]], training_labels)
    print('Time to train RF  with top 5 features{}'.format(time.time() - tme))
    tme = time.time()
    predictions = list(rf_classifier.predict(val_features[:, [14, 12, 11, 18, 15]]))
    print('Time to predict rf 5 {}'.format(time.time() - tme))
    print(f1_score(val_labels, predictions))


def random_search():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)
    rf_random = RandomizedSearchCV(estimator=rf_classifier, param_distributions=random_grid, n_iter=50, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(training_features[:10000], training_labels[:10000])
    print(rf_random.best_params_)
    best_estimator = rf_random.best_estimator_
    best_estimator.fit(training_features, training_labels)
    return best_estimator


def predict_on_test():
    testing_features = build_features(testing_set, test=True)
    # issue predictions
    predictions = list(rf_classifier.predict(testing_features))
    # write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
    predictions = zip(range(len(testing_set)), predictions)
    with open("improved_predictions.csv", "w", newline='') as pred1:
        writer = csv.DictWriter(pred1, fieldnames=["ID", "category"])
        writer.writeheader()
        csv_out = csv.writer(pred1)
        for row in predictions:
            csv_out.writerow(row)


def main():
    load_and_build_data()
    train_and_predict_on_val()
    feature_selection()
    predict_on_test()


if __name__ == 'main':
    main()
