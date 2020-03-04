---
title: 'REINFORCE Trick'
date: 2019-03-04
permalink: /posts/2020/03/reinforce-trick/
tags:
  - Reinforcement Learning
  - Tricks
  - Machine Learning
  - Variational Inference
---

> In this article, I give a brief summary of the so called "REINFORCE" trick.

### Problem

Assume that we have the following items:

1. a random variable $x\sim p(x;\theta)$ where $p$ is a parametric distribution modelled by $\theta$;

2. a function of $x$ which is denoted as $f(x)$.

The quantity we wish to calculate is:
$$
\nabla_\theta \mathbb{E}_{p(x;\theta)}\left[ f(x) \right]
$$

For example, $x$ and $f(x)$ could be:

1. trajectories and reward function in Reinforcement Learning;
2. variational distribution and log-likelihood of joint distribution of observable and latent variables.

### Core Transformation

$$
\nabla_\theta p(x;\theta) = p(x;\theta) \nabla_\theta \log p(x;\theta)
$$

### REINFORCE Trick

The problem of our objective quantity ($\nabla_\theta \mathbb{E}_{p(x;\theta)}\left[ f(x) \right]$) is that the parameter $\theta$ that need to be optimised doesn't exist in $f(x)$, thus we can't get derivative of it with samples drawn from $p(x;\theta)$.

Luckily, we could introduce the following REINFORCE trick to address the problem:

$\nabla_\theta \mathbb{E}_{p(x;\theta)}\left[ f(x) \right] = \nabla_\theta \int f(x) p(x;\theta) dx$

$
 = \int f(x) \nabla_\theta p(x;\theta) dx 
$

$
= \int f(x) p(x;\theta) \nabla_\theta \log p(x;\theta) dx 
$

$= \mathbb{E}_{p(x;\theta)} \left[ f(x)  \nabla_\theta \log p(x; \theta) \right]$

Then, with samples from $p(x; \theta)$, we could approximate the expectation by Monte Carlo method, i.e.

$$
\nabla_\theta \mathbb{E}_{p(x;\theta)}\left[ f(x) \right] \approx \frac{1}{N} \sum^N_{i=1} f(x_i) \nabla_\theta \log p(x_i; \theta)
$$

Now, we could update $\theta$ by calculating $\nabla_\theta \log p(x_i; \theta)$ based on the samples $x_i$.