# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:01:35 2018

@author: assae
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import random


#%% Question 7

#%% Graph

with open("CA-GrQc.txt", 'rb') as file:
    Gi=nx.read_edgelist(file, create_using=nx.DiGraph())
    
G = nx.Graph(Gi)

#%% Nodes/Edges

nb_nodes = nx.number_of_nodes(G)
print(nb_nodes)
nb_edges = nx.number_of_edges(G)
print(nb_edges)
    

#%% Connectivity

print(nx.is_connected(G))
print(nx.number_connected_components(G))

#%% Connectivity

liste_cc = nx.connected_component_subgraphs(G)
liste = []

for g in liste_cc:
    liste.append(nx.number_of_edges(g))

counts = [(x,liste.count(x)) for x in set(liste)]    
countsf = list(zip(*counts))

plt.figure()
plt.xlabel("Size of the connected component")
plt.ylabel("Count")
plt.title("Connectivity of the graph")
plt.loglog(countsf[0], countsf[1], 'bo')


#%% Connectivity : giant connected component

liste_cc = nx.connected_component_subgraphs(G)
gmax = 0
nmax = 0 
for g in liste_cc: 
    if nx.number_of_nodes(g) > nmax:
        gmax=g
        nmax =  nx.number_of_nodes(g)

nb_nodesgcc = nx.number_of_nodes(gmax)
print(nb_nodesgcc)
nb_edgesgcc = nx.number_of_edges(gmax)
print(nb_edgesgcc)


#%% Degree

d = G.degree()
d= dict(d)

#%% Degree: metrics

val = list(d.values())
mini = min(val)
maxi = max(val)
mean = np.mean(np.array(val))
med = np.median(np.array(val))

#%% Power-law distribution

cs = [(x,val.count(x)) for x in set(val)]  
cs = sorted(cs, key=lambda x: x[0])
csf = list(zip(*cs))
    
plt.figure()
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the graph")
plt.plot(csf[0], np.array(csf[1])*(1/5242), 'bo')

plt.figure()
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the graph")
plt.loglog(csf[0], np.array(csf[1])*(1/5242), 'bo')

#%% Parameters of the power-law distribution

x = np.array(csf[0])
y = np.array(csf[1])*(1/5242)

logx = np.log(np.array(csf[0]))
logy = np.log(np.array(csf[1])*(1/5242))

fit = np.polyfit(logx, logy, 1)
fit_fn = np.poly1d(fit) 


yfit = lambda x: np.exp(fit_fn(np.log(x)))


plt.figure()
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the graph")
plt.loglog(x,y, 'bo', x, yfit(x), 'k')


c=np.exp(fit_fn[0])
alpha = -fit_fn[1]
def f(x):
    return(c*(np.exp(-alpha*np.log(x))))
    

plt.figure()
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the graph")
plt.plot(csf[0], np.array(csf[1])*(1/5242), 'bo')
X = list(range(2, 81))
plt.plot(X, f(X), 'k')

#%% Triangles in the GCC

GCC = gmax

nb_triangles = sum(dict(nx.triangles(gmax)).values())
val_triangles = list(dict(nx.triangles(gmax)).values())

cst = [(x,val_triangles.count(x)) for x in set(val_triangles)]  
cst = sorted(cst, key=lambda x: x[0])
cstf = list(zip(*cst))

plt.figure()
plt.xlabel("Number of triangles")
plt.ylabel("Probability")
plt.title("Triangle participation distribution of the graph (Normalized)")
plt.hist(np.array(cstf[1]), 150, density=True)

#%% Triangles in the GCC : spectral approach 

A = nx.to_numpy_matrix(GCC)
eig = sp.linalg.eigh(A, eigvals_only=True)
eigcub = np.array(eig)**3
eigsum = sum(eigcub)
tot_triangles = eigsum/6

#%% Triangles in the GCC : spectral approach : low-rank approximation

liste1=[]
liste2=[]
eigsort = sorted(eigcub, key=lambda x: abs(x), reverse=True)    
for i in range(4157):
    su = sum(eigsort[0:i+1])/6
    liste1.append(i)
    error = abs(su-tot_triangles)/su
    liste2.append(error)

plt.xlabel("Number of retained eigenvalues (in decreasing order of absolute value)")
plt.ylabel("Error in the number of triangles")
plt.title("Error in the number of triangles in function of the retained eigenvalues")
plt.plot(liste1, liste2)



#%% Question 8

#%% Generate the Erdős–Rényi random graph

n = 1000
p = 0.009
seed = 181
G = nx.fast_gnp_random_graph(n, p, seed)

#%% Nodes/Edges

nb_nodes = nx.number_of_nodes(G)
print(nb_nodes)
nb_edges = nx.number_of_edges(G)
print(nb_edges)

#%% Degree

d = G.degree()
d = dict(d)
val = list(d.values())
mean = np.mean(np.array(val))

#%% Connectivity

nx.is_connected(G)

#%% Degree distribution

cs = [(x,val.count(x)) for x in set(val)]  
cs = sorted(cs, key=lambda x: x[0])
csf = list(zip(*cs))
    
plt.figure()
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the random graph")
plt.plot(csf[0], np.array(csf[1])*(1/1000), 'bo')

def poisson(k):
    return np.exp(-mean)*(np.power(mean, k)/np.math.factorial(k))

x= np.arange(0, 20, 1)
plt.plot(x, [poisson(k) for k in x], 'k')




#%% Question 9

#%% Generate the Kronecker graph

a = 0.99
b = 0.26
c = 0.53


initiator = np.array([[a,b],[b,c]])
A = initiator
G = nx.Graph()

nb_iterations = 12
for i in range(nb_iterations):
    A = np.kron(A, initiator)
    
for i in range(len(A)):
    for j in range(len(A)):
        chance = np.random.binomial(1, A[i][j])
        if chance == 1:
            G.add_edge(i,j)
 

    
#%% Basics informations on this graph
    
nb_nodes = nx.number_of_nodes(G)
print(nb_nodes)
nb_edges = nx.number_of_edges(G)
print(nb_edges)

print(nx.is_connected(G))
print(nx.number_connected_components(G))

#%% GCC

liste_cc = nx.connected_component_subgraphs(G)
gmax = 0
nmax = 0 
for g in liste_cc: 
    if nx.number_of_nodes(g) > nmax:
        gmax=g
        nmax =  nx.number_of_nodes(g)

nb_nodesgcc = nx.number_of_nodes(gmax)
print(nb_nodesgcc)
nb_edgesgcc = nx.number_of_edges(gmax)
print(nb_edgesgcc)


#%% Opening the real graph

with open("CA-GrQc.txt", 'rb') as file:
    Gi=nx.read_edgelist(file, create_using=nx.DiGraph())
    
Gr = nx.Graph(Gi)

#%% Degree (Kronecker Graph)

d = G.degree()
d= dict(d)
val = list(d.values())
mean = np.mean(val)

#%% Degree (Real Graph)

dr = Gr.degree()
dr = dict(dr)
valr = list(dr.values())
meanr = np.mean(valr)

#%% Power-law distribution (Kronecker and Real Graph)

cs = [(x,val.count(x)) for x in set(val)]  
cs = sorted(cs, key=lambda x: x[0])
csf = list(zip(*cs))

csr = [(x,valr.count(x)) for x in set(valr)]  
csr = sorted(csr, key=lambda x: x[0])
csfr = list(zip(*csr))
 


x = np.array(csf[0])
y = np.array(csf[1])*(1/nb_nodes)
logx = np.log(np.array(csf[0]))
logy = np.log(np.array(csf[1])*(1/nb_nodes))
fit = np.polyfit(logx, logy, 1)
fit_fn = np.poly1d(fit) 
yfit = lambda x: np.exp(fit_fn(np.log(x)))

xr = np.array(csfr[0])
yr = np.array(csfr[1])*(1/5242)

c=np.exp(fit_fn[0])
alpha = -fit_fn[1]
def f(x):
    return(c*(np.exp(-alpha*np.log(x))))
    


plt.subplots(2,2, figsize=(10, 10))  
plt.subplot(221)
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the Kronecker graph")
plt.plot(csf[0], np.array(csf[1])*(1/nb_nodes), 'bo')
X = list(range(2, 81))
plt.plot(X, f(X), 'k')

plt.subplot(223)
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the Kronecker graph")
plt.loglog(csf[0], np.array(csf[1])*(1/nb_nodes), 'bo')
plt.loglog(x,y, 'bo', x, yfit(x), 'k')
 
plt.subplot(222)
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the real graph")
plt.plot(csfr[0], np.array(csfr[1])*(1/5242), 'bo')
X = list(range(2, 81))
plt.plot(X, f(X), 'k')

plt.subplot(224)
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree distribution of the real graph")
plt.loglog(csfr[0], np.array(csfr[1])*(1/5242), 'bo')
plt.loglog(xr ,yr, 'bo', xr, yfit(xr), 'k')


#%% Connectivity (Kronecker and Real Graph)

liste_cc = nx.connected_component_subgraphs(G)
liste = []
for g in liste_cc:
    liste.append(nx.number_of_edges(g))
counts = [(x,liste.count(x)) for x in set(liste)]    
countsf = list(zip(*counts))

liste_ccr = nx.connected_component_subgraphs(Gr)
lister = []
for g in liste_ccr:
    lister.append(nx.number_of_edges(g))
countsr = [(x,lister.count(x)) for x in set(lister)]    
countsfr = list(zip(*countsr))


plt.subplots(1,2, figsize=(10, 4.5)) 
plt.subplot(121)
plt.xlabel("Size of the connected component")
plt.ylabel("Count")
plt.title("Connectivity of the graph \n (Kronecker Graph)")
plt.loglog(countsf[0], countsf[1], 'bo')

plt.subplot(122)
plt.xlabel("Size of the connected component")
plt.ylabel("Count")
plt.title("Connectivity of the graph \n (Real Graph)")
plt.loglog(countsfr[0], countsfr[1], 'bo')


#%% Clustering coefficient (Kronecker and Real Graph)

avc = nx.average_clustering(G)
avcr = nx.average_clustering(Gr)

clus = list(dict(nx.algorithms.cluster.clustering(G)).values())
d = G.degree()
d= dict(d)
val = list(d.values())
z = list(zip(val, clus))

liste1 = []
liste2 = []
for i in range(40):
    for j in range(len(z)):
        if z[j][0] == i:
            liste1.append(z[j])       
    liste2.append(liste1)
    liste1 = []
 
couples = []    
for i in range(len(liste2)):
    if len(liste2[i])!=0:
        dezip = list(zip(*liste2[i]))
        couples.append((dezip[0][0], np.mean(dezip[1])))

cpls = list(zip(*couples))
    
        
clusr = list(dict(nx.algorithms.cluster.clustering(Gr)).values())
dr = Gr.degree()
dr = dict(dr)
valr = list(dr.values())
zr = list(zip(valr, clusr))


liste1r = []
liste2r = []
for i in range(81):
    for j in range(len(zr)):
        if zr[j][0] == i:
            liste1r.append(zr[j])       
    liste2r.append(liste1r)
    liste1r = []
 
couplesr = []    
for i in range(len(liste2r)):
    if len(liste2r[i])!=0:
        dezipr= list(zip(*liste2r[i]))
        couplesr.append((dezipr[0][0], np.mean(dezipr[1])))

cplsr = list(zip(*couplesr))

plt.subplots(1,2, figsize=(10, 5)) 
plt.subplot(121)
plt.xlabel("Degree")
plt.ylabel("Average clustering coefficient")
plt.title("Clustering coefficient distribution in \n function of the degree (Kronecker graph)")
plt.loglog(cpls[0], cpls[1], 'bo')

plt.subplot(122)
plt.xlabel("Degree")
plt.ylabel("Average clustering coefficient")
plt.title("Clustering coefficient distribution in \n function of the degree \n (Real graph)")
plt.loglog(cplsr[0], cplsr[1], 'bo')




#%% Question 10

#%% Graph

with open("CA-GrQc.txt", 'rb') as file:
    Gi=nx.read_edgelist(file, create_using=nx.DiGraph())
    
G = nx.Graph(Gi)

#%% Getting the GCC

liste_cc = nx.connected_component_subgraphs(G)
gmax = 0
nmax = 0 
for g in liste_cc: 
    if nx.number_of_nodes(g) > nmax:
        gmax=g
        nmax =  nx.number_of_nodes(g)

nb_nodesgcc = nx.number_of_nodes(gmax)
print(nb_nodesgcc)
nb_edgesgcc = nx.number_of_edges(gmax)
print(nb_edgesgcc)

GCC = gmax

#%%

rates = np.arange(0, 0.21, 0.01)
d = G.degree()
d= dict(d)
nb_nodes = nx.number_of_nodes(G)
nodes_list = list(G.nodes)

#First strategy : random deletion

fin_small = []
fin_gcc = []
for r in rates:
    nb_to_remove = int(r*nb_nodes)
    Gr = G.copy()
    nodes_copy = nodes_list.copy()
    
    #Remove a random node
    for j in range(nb_to_remove):
        rem = random.choice(nodes_copy)
        Gr.remove_node(rem)
        nodes_copy.remove(rem)
        
    liste_cc = list(nx.connected_component_subgraphs(Gr))
    gmax = 0
    mmax = 0 
    for h in liste_cc: 
        if nx.number_of_nodes(h) > mmax:
            gmax = h
            mmax = nx.number_of_nodes(h)        
    liste = []
    for g in liste_cc:
        if nx.number_of_edges(g) != mmax:
            liste.append(nx.number_of_edges(g))       
    tot = sum(liste)
    fin_small.append((r, tot))
    fin_gcc.append((r, mmax))
    

#Second strategy : targeted deletion

tar_small = []
tar_gcc = []
for r in rates:
    nb_to_remove = int(r*nb_nodes)
    Gr = G.copy()
    nodes_copy = nodes_list.copy()
    d = Gr.degree()
    d = dict(d)
    val = list(d.values())
    keys = list(d.keys())
    key_val = list(zip(keys, val))
    sorted_list = sorted(key_val, key=lambda x: x[1], reverse = True)
    
    #Remove nodes with highest degree
    for i in range(nb_to_remove):
        Gr.remove_node(sorted_list[0][0])
        sorted_list.remove(sorted_list[0])
    
    liste_cc = list(nx.connected_component_subgraphs(Gr))
    gmax = 0
    mmax = 0 
    for h in liste_cc: 
        if nx.number_of_nodes(h) > mmax:
            gmax = h
            mmax = nx.number_of_nodes(h)        
    liste = []
    for g in liste_cc:
        if nx.number_of_edges(g) != mmax:
            liste.append(nx.number_of_edges(g))       
    tot = sum(liste)
    tar_small.append((r, tot))
    tar_gcc.append((r, mmax))
 

#%% Plot the collected data

plt.figure(figsize=(10,10))
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
final_small = list(zip(*fin_small))
final_gcc = list(zip(*fin_gcc))
target_small = list(zip(*tar_small))
target_gcc = list(zip(*tar_gcc))
p1 = ax.plot(final_gcc[0], final_gcc[1], 'bo', 
             label="Size of the GCC evolving with random deletion" ) 
p2 = ax.plot(final_small[0], final_small[1], 'b+', 
             label="Size of the rest of components evolving with random deletion")
p3 = ax.plot(target_gcc[0], target_gcc[1], 'ro', 
             label="Size of the GCC evolving with targeted deletion")
p4 = ax.plot(target_small[0], target_small[1], 'r+', 
             label="Size of the rest of components evolving with targeted deletion")
plt.xlabel("Fraction of deleted nodes")
plt.ylabel("Size of connected components")
plt.title("Size of GCC and of the rest of connected components \n over the fraction of deleted nodes")
ax.legend(loc=8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))

        
        
        





