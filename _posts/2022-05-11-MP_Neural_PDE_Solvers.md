---
layout: post
title: Message Passing Neural PDE Solvers
published: true
---
![_config.yml]({{ site.baseurl }}/images/MP-PDE-Solver.png){: .align-center}

<p>  <div style="text-align: justify"> <em> This blog post is based on the paper: Brandstetter, Johannes, Daniel Worrall, and Max Welling. "Message passing neural PDE solvers." arXiv preprint arXiv:2202.03376 (2022).  </em> </div> </p>


## Why Deep Learning for physics simulations?
<p>  <div style="text-align: justify"> Many problems in science and engineering (Fluid mechanics, plasticity, quantum mechanics, climate models, biological systems...) are described using Partial Differential Equations (PDE). Most PDEs described real world problems require numerical solutions. For large problems the numerical solutions are computationally expensive. Well established methods to solve PDEs are the Finite Difference Method (FDM), Finite Volume Method (FVM) and Finite Element Method (FEM). These methods require a fine spatial discretization of the computational domains and get very slow and inefficient. Many scientists have been trying to overcome these difficulties in classical numerical methods by leveraging Machine Learning (ML) methods. Recently a lot of efforts have been shifted towards Deep Learning (DL) and Graph Neural Networks (GNN). We will kick of this blog post by introducing the the general setting of PDE based numerical simulations followed by a brief introduction to  DL and GNNs and then focus on the solution of PDEs using DL and GNNs. If you are familiar with PDEs and/or GNNs you can skip the next two sections and proceed with the exciting part (I would argue that going back to some basics from time to time is also exciting).  </div> </p>

Many problems in physics and engineering can be described using PDEs. For an arbitrary problem the PDE can be written as

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/PDE.png){: .align-center}
{: refdef}

with the initial and boundary conditions 
{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/initial_condition.png){: .align-center}
{: refdef}

<p>  <div style="text-align: justify">
In simple words: The evolution of a variable state in time is described by the value of the variable at a given time and its spatial derivatives.
An example of a PDE is the heat equation. The heat equation describes how quantities (such as heat) diffuse through a given region. To solve the PDE describing the problem using classical numerical methods (FEM, FVM; FDM...) an initial condition describing the disctribution of the temperature at time t=0 has to be provided. The spatial derivatives are then approximated using discretization schemes and the time derivatives are computed using time stepping schemes such as the Euler method or the Runge Kutta (RK) method. In the following we illustrate the solution of the heat equation with two different initial conditions for multiple time steps.
</div> </p>

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/Diffusion.png){: .align-center}
{: refdef}

<p>  <div style="text-align: justify">
To motivate the need to predict the solutions of PDEs it is helpful to have a look at another type of PDEs to illustrate the variety of problems they describe. Looking at different types of PDEs will be helpful later during the discussion of the paper, where you will see the importance of ML algorithms that can generalize to different PDE types.
Let's have a look at the advection euqation (a PDE of hyperbolic type). The advection equation models how a quantity is transported in space: Examples include the wave propagation of a tsunami (The shalow water equation), the contaminant transport in a river by fluid motion... In the following video you can see how the numerical solution of the shallow water euqation describes the propagation of a wave in a fluid with reflective boundary conditions. 
  
{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/ShallowWater_PDE.png){: .align-center}
{: refdef}  

The above mentionned problems take only a couple of second to solve for one initial parameter. If you think if real world problems, these are usually a lot larger and are not one dimensional. Thus simulations are computationally expensive and the resulting output is also very high dimensional. We usually talk about several million values (nodes) for each timestep. This doesn't seem large enough? Here are two of the largest  numerical simulations of PDEs ever produced:
</div> </p>

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/turbulence_showcase.png){: .align-center}

2) https://arxiv.org/pdf/1607.00630.pdf
{: refdef}

{:refdef: style="text-align: center;"} 
![_config.yml]({{ site.baseurl }}/images/space_showcase.jpg){: .align-center}

1) https://singularityhub.com/2021/09/17/the-biggest-simulation-of-the-universe-yet-stretches-back-to-the-big-bang/
{: refdef} 


Now that I assume this digression thrilled you and that you recognize the urge for efficient prediction methods for simulations, we go back to theory and let's talk Machine Learning 
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
