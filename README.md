# BoCF
This repository provides the official implimentation of the illuminant estimation algorithm **BoCF** proposed in paper *[Bag of Color Features For Color Constancy](https://ieeexplore.ieee.org/document/9130881)*  using *[INTEL-TAU dataset](http://urn.fi/urn:nbn:fi:att:f8b62270-d471-4036-b427-f21bce32b965)*. 

# Introduction
In this paper, we propose a novel color constancy approach, called **BoCF**, building upon Bag-of-Features pooling. The proposed method substantially reduces the number of parameters needed for illumination estimation. At the same time, the proposed method is consistent  with the color constancy assumption stating that global spatial information is not relevant for illumination estimation and local information (edges, etc.) is sufficient. 

<img src="figures/intro22.jpg" width="900">

Furthermore, **BoCF** is consistent with color constancy statistical approaches and can be interpreted as a learning-based generalization of many statistical approaches. To further improve the illumination estimation accuracy, we propose a novel attention mechanism for the **BoCF** model with two variants based on self-attention.
