---
layout: post
title: Message Passing Neural PDE Solvers
published: true
---
![_config.yml]({{ site.baseurl }}/images/MP-PDE-Solver.png)

<p>  <div style="text-align: justify"> <em> This blog post is based on the paper: Brandstetter, Johannes, Daniel Worrall, and Max Welling. "Message passing neural PDE solvers." arXiv preprint arXiv:2202.03376 (2022).  </em> </div> </p>


## Why Deep Learning for physics simulations?
<p>  <div style="text-align: justify"> Many problems in science and engineering (Fluid mechanics, plasticity, quantum mechanics, climate models, biological systems...) are described using Partial Differential Equations (PDE). Most PDEs described real world problems require numerical solutions. For large problems the numerical solutions are computationally expensive. Well established methods to solve PDEs are the Finite Difference Method (FDM), Finite Volume Method (FVM) and Finite Element Method (FEM). These methods require a fine spatial discretization of the computational domains and get very slow and inefficient. Many scientists have been trying to overcome these difficulties in classical numerical methods by leveraging Machine Learning (ML) methods. Recently a lot of efforts have been shifted towards Deep Learning (DL) and Graph Neural Networks (GNN). We will kick of this blog post by introduction the the general setting of PDE based numerical simulations followed by a brief introduction to  DL and GNNs and then focus on the solution of PDEs using DL and GNNs.  </div> </p>

Many problems in physics and engineering can be described using PDEs. For an arbitrary problem the PDE can be written as
$$
\frac{\partial \mathbf{u}}{\partial t } = L(u,t)
$$


## Deep Learning
### Deep Learning and Artificial Neural Networks
here we introduce briefly ANNs and DL 
### Convolutional Neural Networks
A brief introduction in CNNs, mostly using plots/graphs is introduced here. Not much formulas. Just for the general undestanding
### Graph Neural Networks
The concept of graph embedding, aggregation and message passing.
Many good resources to present this with graphs. Also from the main paper of the blog.

## Deep Learning for physics simulations: The 2 paradigms 
### Neural Opeators and Autoregressive Methods

## GNNs and FNOs

## What's new in the work of Brandstetter et al.
In this section we present (an also discuss?) the "cool" tricks making this paper special
### The pushforward trick
### Temporal Bundling
### Generalisation over different classes of PDEs

## How well do MP-N-PDE Solvers perform?
This is the part with numerical results from this paper. Even though many methods are presented, only results from Brandstetter et al. are discussed.


## What's left to do?
here critique and a small review of the presented methods can be written.

## Conclusion
