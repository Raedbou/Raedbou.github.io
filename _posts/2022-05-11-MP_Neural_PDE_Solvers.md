---
layout: post
title: Message Passing Neural PDE Solvers
published: true
---
![_config.yml]({{ site.baseurl }}/images/MP-PDE-Solver.png){: .align-center}

<p>  <div style="text-align: justify"> <em> This blog post is based on the paper: Brandstetter, Johannes, Daniel Worrall, and Max Welling. "Message passing neural PDE solvers." arXiv preprint arXiv:2202.03376 (2022). </em> </div> </p>


<p>  <div style="text-align: justify"> Many problems in science and engineering (Fluid mechanics, plasticity, quantum mechanics, climate models, biological systems...) are described using Partial Differential Equations (PDE). Most PDEs described real world problems require numerical solutions. For large problems the numerical solutions are computationally expensive. Well established methods to solve PDEs are the Finite Difference Method (FDM), Finite Volume Method (FVM) and Finite Element Method (FEM). These methods require a fine spatial discretization of the computational domains and get very slow and inefficient. Many scientists have been trying to overcome these difficulties in classical numerical methods by leveraging Machine Learning (ML) methods. Recently a lot of efforts have been shifted towards Deep Learning (DL) and Graph Neural Networks (GNN). We will kick of this blog post by introducing thegeneral setting of PDE based numerical simulations followed by a brief introduction to  DL and GNNs and then focus on the solution of PDEs using DL and GNNs. If you are familiar with PDEs and/or GNNs you can skip the next two sections and proceed with the exciting part (I would argue that going back to some basics from time to time is also exciting).  </div> </p>

## Numerical simulation of PDEs
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
![_config.yml]({{ site.baseurl }}/images/diffusion_PDE.png){: .align-center}
{: refdef}

<p>  <div style="text-align: justify">
To motivate the need to predict the solutions of PDEs it is helpful to have a look at another type of PDEs to illustrate the variety of problems they describe. Looking at different types of PDEs will be helpful later during the discussion of the paper, where you will see the importance of ML algorithms that can generalize to different PDE types.
Let's have a look at the advection euqation (a PDE of hyperbolic type). The advection equation models how a quantity is transported in space: Examples include the wave propagation of a tsunami (The shalow water equation), the contaminant transport in a river by fluid motion... In the following example you can observe the numerical solution of the shallow water euqation  (that e.g. describes the propagation of a wave in a fluid) with reflective boundary conditions. 
</div> </p>

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/ShallowWater_PDE.png){: .align-center}
{: refdef}  

<p>  <div style="text-align: justify">
The above mentionned problems take only a couple of second to solve for one initial parameter. If you think if real world problems, these are usually a lot larger and are not one dimensional. Thus simulations are computationally expensive and the resulting output is also very high dimensional. We usually talk about several million values (nodes) for each timestep. This doesn't seem large enough? Here are two of the largest  numerical simulations of PDEs ever produced:
</div> </p>

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/Showcase_PDE.jpg){: .align-center}
{: refdef}
[**(left)**](https://arxiv.org/pdf/1607.00630.pdf) Largest turbulence simulation: 1.014e+12 nodes on 65000 cores: [Video](https://www.mso.anu.edu.au/~chfeder/pubs/sonic_scale/Federrath_sonic_scale_lowres.mp4)

[**(Right)**](https://www.nao.ac.jp/en/news/science/2021/20210910-cfca.html) Largest simulation of the Universe: 2.1e+12 particles on 40200 cores: [Video](https://www.youtube.com/watch?time_continue=3&v=R7nV6JEMGAo&feature=emb_title)

<p>  <div style="text-align: justify">
Now that I assume this digression thrilled you and that you recognize the urge for efficient prediction methods for simulations, we dive back into theory and have a look at some DL concepts. 
</div> </p>

## Deep Learning basics

### Artificial Neural Networks
  <p>  <div style="text-align: justify">
Deep Leaning is a category of machine learning designed to predict an output Y  based on an input X. The basic model used in Deep Learning are Artificial Neural Networks. Artificial Neural Netowrks learn patterns between X and Y and use those patterns to predict output for new unseen input variables X. They can be viewed as weighted directed graphs. In Artificial Neural Networks the nodes are called neurons and the weighted edges are called connections. An Artificial Neural Network receives an input X,eg. Vector, Matrix... and the elements of the inputs are multiplied each by the weight of the connection it is connected with. All the weighted sums are respectively summed up at a neuron that is connected to the input nodes and the weighted sum is passed to a function called the activation function. The activation function maps the weighted sum to the output variables. This results in the following relation between input, output, weights w and activation function f.
</div> </p>
{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/Perceptron.png){: .align-center}
{: refdef}
<p>  <div style="text-align: justify">
When an Artificial Neural Network has more then one layer of neurons between the input and the output we speak of Multilayer Perceptrons. The first layer is the input layer, the last layer is the output layer and the layers in between are referred to as hidden layers. In the case of the example below imagine how the information is passed from left to right (from input to output as shown by the errors) throughout the layers: The network is called a Feedforward Network. The above shown relationship between input values, weights and activation function is applied at each node of each layer.
</div> </p>
{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/ANN.png){: .align-center}
{: refdef}

<p>  <div style="text-align: justify">
Similar to a linear regressor the task here is to find the optimal set of weights that best expresses the input/output dependencies. This is an opimisation task that can be solved by minimizing a loss function with respect to the weights. 

Here I link blog posts that I find very helpful in understanding some of the above mentioned concepts and tools:
</div> </p>

  - [Activation functions](https://towardsdatascience.com/everything-you-need-to-know-about-activation-functions-in-deep-learning-models-84ba9f82c253)
  
  - [Loss functions](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)
  
  - [Optimization algorithms](https://towardsdatascience.com/understanding-optimization-algorithms-in-machine-learning-edfdb4df766b) ( and if you are more into reading papers, you might enjoy reading this [survey](https://arxiv.org/pdf/1906.06821.pdf))

### Graph Neural Networks and Message Passing
In their work, Brandstetter et al. argue that Graph Neural Networks and the concept of Message Passing satisfy conditions that are required in numerical solvers and that several previously developed ML based solvers do not satisfy. The most relevant condition for engineering applications is generalisability. Generalisability over different resolutions, topologies, geometries, initial conditions and dimensions can give rise to generic solvers over several application domains. To understand the foundation behind this argument, let's understand the basics of Graph Neural Networks.

Many real data cannot be respresented in a way that can be processed using ANNs (and furhter architectures such as CNNs). However, most data can be described as graphs. Graphs consist of a list of objects with connectios between them. Graph alike structures can be found in many physical problems: 
  - In structural dynamics the objects can be certain domains/points of a geometry and the connection is the distance between them or certain material properties such as damping and stiffness. 
  - In astrophysics objects can be planets/stars and the connections can be distances, forces describing gravitatin, masses ...
  - In gas and fluid dynamics particles can seen as objects and the connections are distances between these particles and further material properties such as density and viscosity.
  - In molecular dynamics the objecs are atoms described by an atom type and connected by relative positions and edge types.  

The objects in a graph are called nodes and the connections are edges. 
{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/Graph.png){: .align-center}
{: refdef}  

Nodes can have vector attributes and edges can contain several information packed in a vector. Edge information doesn't necessarly have to include computed edges such as distances. It can e.g. include information on neighbouring nodes instead. This gives the flexibility of choosing how many neighbouring nodes can influence a given state. If you now think back about the generalisability assumpution you might notice predicting a state by the state of only it's neighbouring state doesn't necessarly include the notion of a "global geometry/tobology" and thus it enforces generalisability over different topologies and geometries. It also supports the generalisation over different resolutions and discretizations argument: Instead of of predicting states using a fixed amount of data points in a geometry, the state of each node can be predicted using the states of a given number of neighbours and the geometry can still be embedded into the edge attributes.

Now that the data is embedded in graph representations, how do we learn on graphs? The answer to this question is Graph Neural Networks.

Graph Neural Networks are built using the Message Passing framework introduced by [Giler et al.](https://arxiv.org/abs/1704.01212). They adopt a graph-in graph-out principle. This means that for graph inputs with nodes, edge and global context information the output is a graph with the same connectivity as the input graph. The idea of message passing boils down to three main steps:

  1) For every node of the graph as message is computed for all neighbouring nodes. A messages is a function of the node information, the neighbor information, and the edge between them. This function can be a Multilayer Perceptron.

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/MP1.png){: .align-center}
{: refdef}  

  2) After all mesages are computed each node aggregates the messages it receives from its neighbours. Aggregation is the application of a permutation invariant function. An aggregation function can be e.g. a sum or an average, but is not limited to such simple operations.
 
 {:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/MP2.png){: .align-center}
{: refdef}  
  
  3) After all messages from neighbours are passed to the current node, the node state is updated. The update is a function of the current node state and the aggregated messages (and possibly other information). The update function is learned using training data and can e.g. be a Feedfowrd Neural Network.

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/MP3.png){: .align-center}
{: refdef}  

For more detailed information supported with visual interactive material about Graph Neural Networks you can check "[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)". W.L. Hamiltons McGill [course on Graph Representation Learning](https://cs.mcgill.ca/~wlh/comp766/files/chapter4_draft_mar29.pdf) also offers a thorough introduction to GNNs and Message Passing. 


## Connecting the dots

At this points let's connect the key points we learned so far: In simulations we predict future sttes from current states. Similar to simulation, in Message Passing GNNs the transformation of graphs between different states are learned and future states of graph nodes can be predicted from current states. Let's visualize this analogy:

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/MP_PDE_Analogy.png){: .align-center}
{: refdef}  

We saw that simulations can be very expensive and motivated the need for methods that accelerate them. Here Machine Learning comes in handy. GNNs learn efficiently on graphs and many physical data can be interpreted into graphs. This synergy results in Message Passing Neural PDEs solvers.

There are two main paradigms that are been applied in using Machine Learning to solve PDEs: Neural Operators and Autoregressive Methods.

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/NeuralMethods.png){: .align-center}
{: refdef}  

Neural Operators learn predictions from an initial condition to a given timestep t. Neural Operators are trained on example solutions of a given equation and are thus locked to a given equation and do not generalize well. Autoregressive Methods are iterative. They rely on an initial state of a time dependant problem and then iteratively generate solutions from predicted states. In other words, if an Autoregressive Method receives an initial state of a problem and a number of timesteps it generates the prediction at time one from the initial condition, than the prediction at time 2 from the previously made prediction 1 and so forth until n predictions are generated.

## Autoregressive Methods and Instability
In their paper, [Brandstetter et al.](https://arxiv.org/abs/2202.03376) argue that Autoregressive Methods are hard to train. This is due to their instability. The definition of stability in this context is similar to its definition in numerical analysis: Numerical stability concerns how errors introduced during the execution of an algorithm affect the result. In Autoregressive Neural PDE solvers instability means that prediction error get accumulated through the prediction and that for a growing prediction horizon the error explodes.Due to the latter phenomena predictions for large time periods may not be possible. In the following figure we illustrate how error grows in time for autoregressive methods and the prediction diverges from the solution manifold. 

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/Unstable.png){: .align-center}
{: refdef}  

This phenomena has been shown in many applications and has been addressed in many different ways. For e.g. [Kneifl et al.](https://arxiv.org/pdf/2110.13583.pdf) observed error accumulation in an autoregressive framework. [Liu et al.](https://arxiv.org/pdf/2008.09768.pdf) and [Vlachas et al.](https://www.nature.com/articles/s42256-022-00464-w.pdf?origin=ppub) addressed the  problem of instability in autoregressive methods. However not many publications can be found that address instablity of Message Passing Neural PDE solver. That it what makes the paper we discuss here unique. The main contribution of the authors lays in finding "tricks" to improve the stability and the generalisability of Message Passing Autoregressive methods.

## Tricks to impose stability
### The pushforward trick
The authors approach gthe problem in probablistic terms. From a probablistic point of view, the solver is learning the pushforward operator that transforms the probability distribution of a state k to a probability distribution of a state k+1. This means that after one step of autoregression the solver gets as input the probability distribution generated by the learned pushfrward operator. That was the smart way to say that for each time iteration the solver seach a result that constitutes of the ground truth state corrupted by some error. That results in a probability distribution that is shifted. The authors idea is to account for the distribution shift in an axtra loss term of adverserial style. The perturbation is chosen 

This "complicated" approach was implemented with an easy trick. The authors found out that the pushforward trick can be achieved by by unrolling the solver over 2 steps and only bacpropagating the error on the last step. Theyr numerical experiments also showed that this trick makes the training faster and the prediction more stable.

### Temporal Bundling
Another trick effective for stability is predicting multiple time steps at a time. The authors argument that a fewer number of calls of the solver results in fewer distribution shifts which reduces the error propagation. Another possible explanation to the increased stability through temporal bundling is that by predicting over multiple time steps the prediction error is "distributed" over multiple timesteps and the error contribution to single steps is limited.

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/bundling.png){: .align-center}
{: refdef}  

## Training workflow

In the paper the authors formulate the PDE problem

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/pde_formulation.png){: .align-center}
{: refdef}  

This formulation of the problem enables generalization to different PDE classes and type. By changing the parameters β, γ and α different 1D-PDEs are adressed at a time. For example the following (α,β,γ) combinations yield the following PDEs:

<div class="datatable-begin"></div>
|  (α,β,γ)    | Equation    | PDE Type |
| ----------- | ----------- | ----------- |
| (0,η,0)      | Heat euqation     | Parabolic    |
| (0.5,η,0)    | Generalized Burgers equation      | Hyperbolic for η>0   |
| (3,0,1)  | Korteweg–De Vries equation  | Hyperbolic |
<div class="datatable-end"></div>

The choice of 

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/periodic_bc.png ){: .align-center}
{: refdef}  

as a periodic condition guarantees periodicity of the initial condtions and also forcing term. And the parametrisation enables straightforward embedding of the initial condition and forcing term as features.


## How well do MP-N-PDE Solvers perform?
This is the part with numerical results from this paper. Even though many methods are presented, only results from Brandstetter et al. are discussed.


## What's left to do?
here critique and a small review of the presented methods can be written.

## Conclusion
