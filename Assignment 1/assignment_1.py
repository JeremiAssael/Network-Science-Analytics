# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:01:35 2018

@author: assae
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp


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
    liste.append(nx.number_of_nodes(g))

counts = [(x,liste.count(x)) for x in set(liste)]    
countsf = list(zip(*counts))

plt.figure()
plt.xlabel("Size of the connected component")
plt.ylabel("Count")
plt.title("Connectivity of the graph")
plt.plot(countsf[0], countsf[1])
plt.figure()
plt.xlabel("Size of the connected component")
plt.ylabel("Count")
plt.title("Connectivity of the graph: CC with size below 14")
plt.plot(countsf[0][0:len(countsf[0])-1], countsf[1][0:len(countsf[0])-1])


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
plt.loglog(csf[0][0:len(csf)-1], np.array(csf[1][0:len(csf)-1])*(1/5242), 'bo')

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

#%% Generate the Erdős–Rényi random graph















