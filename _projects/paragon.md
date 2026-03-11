---
layout: post
title: PARAGON – Parallel Graph Engine
date: 2026-02-26 05:00:00
description: A high-performance C++ parallel graph processing engine for large-scale graph analytics
# img: assets/img/graph.jpg

importance: 1
category: work
pseudocode: true
---

## Overview

**PARAGON (Parallel Graph Engine)** is a **high-performance graph processing framework written in C++** designed to efficiently execute large-scale graph algorithms using **thread-level parallelism on multicore CPUs**.

The engine focuses on **performance, scalability, and modularity**, enabling efficient execution of classical graph algorithms such as:

* Breadth First Search (BFS)
* PageRank (Push / Pull variants)
* Connected Components
* Single Source Shortest Path (SSSP)
* Triangle Counting

PARAGON uses **static work partitioning, cache-aware data structures, and barrier synchronization** to maximize CPU utilization while minimizing synchronization overhead.

GitHub Repository:

**https://github.com/saketjha34/PARAGON**

---

## Motivation

Large real-world graphs appear in many domains:

* Social networks
* Web graphs
* Transportation networks
* Knowledge graphs
* Biological interaction networks

These graphs often contain **millions or billions of edges**, making sequential algorithms inefficient.

PARAGON was developed to explore **parallel graph processing techniques** similar to systems like:

* GraphLab
* Ligra
* Gunrock

but implemented as a **lightweight high-performance C++ engine**.

---

## System Architecture

The engine follows a **modular architecture**:

```
PARAGON
│
├── Graph Representation
│     ├── Adjacency List
│     └── Compressed Edge Storage
│
├── Parallel Engine
│     ├── Thread Pool
│     ├── Work Partitioning
│     └── Barrier Synchronization
│
├── Algorithms
│     ├── BFS
│     ├── PageRank
│     ├── Connected Components
│     ├── SSSP
│     └── Triangle Counting
│
└── Utilities
      ├── Timer
      ├── Benchmarking
      └── Graph Loader
```

---

## Graph Representation

The graph is stored using a **compressed adjacency list** representation.

Advantages:

* Efficient memory usage
* Fast traversal
* Cache-friendly access patterns

Example structure:

```cpp
class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int n) {
        V = n;
        adj.resize(n);
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
    }
};
```

---

## Parallel Execution Model

PARAGON distributes graph vertices across **multiple worker threads**.

Each thread processes a **partition of vertices** independently.

Key concepts:

* Static partitioning
* Barrier synchronization
* Lock-free reads where possible

Parallel execution skeleton:

```cpp
#pragma omp parallel for schedule(static)
for(int v = 0; v < V; v++){
    process_vertex(v);
}
```

This allows the engine to scale efficiently on **multi-core CPUs**.

---

# Algorithms Implemented

---

# 1. Breadth First Search (BFS)

BFS computes the shortest path distance from a source node in **unweighted graphs**.

### Parallel BFS Pseudocode

```
procedure PARALLEL_BFS(source)

distance[source] = 0
frontier = {source}

while frontier not empty
    next_frontier = {}

    parallel for each vertex u in frontier
        for each neighbor v of u
            if distance[v] == INF
                distance[v] = distance[u] + 1
                add v to next_frontier

    frontier = next_frontier
```

### C++ Implementation

```cpp
void bfs(Graph& g, int source) {
    vector<int> dist(g.V, -1);
    queue<int> q;

    dist[source] = 0;
    q.push(source);

    while(!q.empty()) {
        int u = q.front(); q.pop();

        for(int v : g.adj[u]) {
            if(dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
}
```

---

# 2. PageRank (Push / Pull)

PageRank measures **node importance in directed graphs**.

Formula:

```
PR(v) = (1 - d)/N + d * Σ(PR(u)/deg(u))
```

where:

* `d` = damping factor
* `N` = number of vertices

### Parallel PageRank Pseudocode

```pseudocode
% This quicksort algorithm is extracted from Chapter 7, Introduction to Algorithms (3rd edition)
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\PROCEDURE{Quicksort}{$$A, p, r$$}
    \IF{$$p < r$$}
        \STATE $$q = $$ \CALL{Partition}{$$A, p, r$$}
        \STATE \CALL{Quicksort}{$$A, p, q - 1$$}
        \STATE \CALL{Quicksort}{$$A, q + 1, r$$}
    \ENDIF
\ENDPROCEDURE
\PROCEDURE{Partition}{$$A, p, r$$}
    \STATE $$x = A[r]$$
    \STATE $$i = p - 1$$
    \FOR{$$j = p$$ \TO $$r - 1$$}
        \IF{$$A[j] < x$$}
            \STATE $$i = i + 1$$
            \STATE exchange
            $$A[i]$$ with $$A[j]$$
        \ENDIF
        \STATE exchange $$A[i]$$ with $$A[r]$$
    \ENDFOR
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```

### C++ Snippet

```cpp
for(int iter = 0; iter < iterations; iter++) {

#pragma omp parallel for
    for(int v = 0; v < V; v++)
        new_pr[v] = (1 - d) / V;

#pragma omp parallel for
    for(int u = 0; u < V; u++) {
        for(int v : adj[u]) {
            new_pr[v] += d * pr[u] / adj[u].size();
        }
    }

    pr.swap(new_pr);
}
```

---

# 3. Connected Components

The algorithm identifies **clusters of connected vertices**.

### Pseudocode

```
initialize label[v] = v

repeat
    parallel for each edge (u,v)
        label[v] = min(label[v], label[u])
until convergence
```

This technique is similar to **label propagation algorithms** used in distributed graph systems.

---

# 4. Single Source Shortest Path (SSSP)

For weighted graphs PARAGON implements a **parallel relaxation-based algorithm**.

### Pseudocode

```
distance[source] = 0

repeat
    parallel for each edge (u,v)
        if dist[u] + w(u,v) < dist[v]
            dist[v] = dist[u] + w(u,v)
until no updates
```

---

# 5. Triangle Counting

Triangle counting identifies **3-node fully connected subgraphs**.

Applications:

* Community detection
* Social network analysis
* Graph clustering

### Pseudocode

```
count = 0

parallel for each vertex u
    for each pair (v,w) in neighbors(u)
        if edge(v,w) exists
            count++
```

---

# Performance Optimizations

Several techniques were used to improve performance:

### Cache-aware adjacency storage

Reduces memory latency.

### Static vertex partitioning

Avoids dynamic scheduling overhead.

### Barrier synchronization

Ensures algorithm correctness across iterations.

### Lock-free reads

Reduces contention between threads.

---

# Example Benchmark

Example graph sizes used for testing:

| Graph           | Nodes | Edges |
| --------------- | ----- | ----- |
| Social Network  | 100K  | 1M    |
| Web Graph       | 500K  | 5M    |
| Synthetic Graph | 1M    | 10M   |

The engine demonstrated **significant speedups on multicore processors** compared to sequential implementations.

---

# Technologies Used

* **C++**
* Multithreading (OpenMP / std::thread)
* Graph Algorithms
* Performance Optimization
* Parallel Computing

---

# Future Improvements

Planned improvements include:

* Dynamic graph support
* GPU acceleration
* Distributed graph processing
* Graph neural network integration

---

# Repository

GitHub:

**https://github.com/saketjha34/PARAGON**

The repository contains:

* modular graph engine
* algorithm implementations
* benchmark examples
* documentation