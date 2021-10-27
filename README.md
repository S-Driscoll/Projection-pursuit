Kurtosis-based Projection Pursuit
======

Kurtosis-based projection pursuit analysis (PPA) is an exploratory data analysis algorithm originally developed by Siyuan Hou and Peter Wentzell in 2011 and remains an active research area for the [Wentzell Research Group](http://groupwentzell.chemistry.dal.ca/) at Dalhousie University. Instead of using variance and distance-based metrics to explore high-dimensional data (PCA, HCA etc.), PPA searches for interesting projections by optimizing kurtosis. This repository contains MATLAB and Python code to perform PPA, published literature involving the on-going development of PPA, as well as some examples of how to apply PPA to uncover interesting projections in high-dimensional data. Below is a figure from our recent paper that I think demonstrates the value of searching for distributions with low kurtosis values.

<h1 align="center">
<img src="https://S-Driscoll.github.io/img/dist.png" alt="kurtosis" width="400"/>
</h1>


MATLAB and Python Functions 
----------

* `projpursuit.m` is a MATLAB function to perform kurtosis-based projection pursuit. It has the following MATLAB function call format:
```matlab
[T, V, ppout] = projpursuit(X,varargin)
```
where X is the m (samples) x n (response variables) data matrix, T is the m x p scores for the samples in each of the p dimensions (default p= 2), V is the corresponding n x p projection vectors and ppout (1 x p) contains the final kurtosis value for each dimension.
* `projpursuit.py` is a Python function that is more or less a line by line port of the MATLAB function. `kurtosis.py` is a python implementation of the MATLAB function `kurtosis.m`. A list of dependencies to run `projpursuit.py` are found in `dependencies.txt`. The Python PPA function has the following call format:
```python
projpursuit(X, **kwargs)
```
that returns T, V, and ppout.

Literature
----------

* [Fast and simple methods for the optimization of kurtosis used as a projection pursuit index (2011)](https://doi.org/10.1016/j.aca.2011.08.006)
* [Re‐centered kurtosis as a projection pursuit index for multivariate data analysis (2013)](https://doi.org/10.1002/cem.2568)
* [Regularized projection pursuit for data with a small sample-to-variable ratio (2014)](https://link.springer.com/article/10.1007/s11306-013-0612-z)
* [Procrustes rotation as a diagnostic tool for projection pursuit analysis (2015)](https://doi.org/10.1016/j.aca.2015.03.006)
* [Projection pursuit and PCA associated with near and middle infrared hyperspectral images to investigate forensic cases of fraudulent documents (2017)](https://doi.org/10.1016/j.microc.2016.10.024)
* [Sparse Projection Pursuit Analysis: An Alternative for Exploring Multivariate Chemical Data (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.9b03166)

Examples
----------

### Wood Identification using Near-infrared (NIR) Spectroscopy and univariate PPA
PPA was originally developed for searching high-dimensional chemical data for informative projections. As such, this example employs a data set designed for the identification of different Brazilian wood species using NIR spectroscopy. The original paper and the data for this example can be found here: [Implications of measurement error structure on the visualization of multivariate chemical data: hazards and alternatives (2018)](https://www.nrcresearchpress.com/doi/abs/10.1139/cjc-2017-0730#.XkHstSMpCCo).

The NIR wood data set contains 4 replicate scans of the follow wood samples: 26 of crabwood, 28 of cedar, 29 of curupixa, and 25 of mahogany. This results 432 samples across 100 NIR channels. Let's apply PCA and PPA and plot the corresponding scores:

```matlab
Xm = X - mean(X);
% PCA of mean centered data via singular value decomposition
[U, S, V] = svds(Xm, 2);
T_PCA = U*S;

% PPA of mean centered data with dim = 2 and number of initial guesses equal to 1000
[T_PPA] = projpursuit(Xm, 2, 1000)

% Class vector
class = class_rep;

% Colour vector for plotting
cvec = ['b' 'k' 'r' 'g'];

% Plot the PCA scores
figure
for i=1:length(T_PCA)
   scatter(T_PCA(i,1), T_PCA(i,2), 300, cvec(class(i)), 'filled', 'MarkerEdgeColor','black') 
    hold on
end
set(gca,'linewidth',2,'FontSize',14)
xlabel('PCA Score 1')
ylabel('PCA Score 2')

% Plot the PPA scores
figure
for i=1:length(T_PPA)
   scatter(T_PPA(i,1),T_PPA(i,2),300,cvec(class(i)),'filled','MarkerEdgeColor','black') 
    hold on
end
set(gca,'linewidth',2,'FontSize',14)
xlabel('PP Score 1')
ylabel('PP Score 2')
```

![PCA vs PPA](https://github.com/S-Driscoll/Projection-pursuit/blob/master/common/images/wood.PNG)

### Unsupervised Facial Recognition using Univariate PPA
Of course, the data being explored does not have to be chemical in nature... the PPA framework can be applied to any multivariate data set. In this example, we will apply it to a subset of [The AT&T face data set](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/). This subset consists of 4 classes (people) each with 10 different grayscale images of their face (112 x 92 pixels). All images were vectorized along the row direction (112 x 92 --> 1 x 10304) producing a 40 x 10304 data set X which was then column mean-centered. Let's apply PCA and PPA and plot the first two scores vectors:

```matlab
Xm = X - mean(X);
% PCA of mean centered data via singular value decomposition
[U, S, V] = svds(Xm, 2);
T_PCA = U*S;

% PPA of mean centered data using PCA compression (15 components) to avoid PPA finding spurious low kurtosis
[U, S, V] = svds(Xm, 15);
[T_PPA] = projpursuit(U*S,2)

% Class vector (classes were vertically stacked in groups of 10)
class = [ones(1,10), 2*ones(1,10), 3*ones(1,10), 4*ones(1,10)];

% Colour vector for plotting
cvec = ['b' 'k' 'r' 'g'];

% Plot the PCA scores
figure
for i=1:length(T_PCA)
   scatter(T_PCA(i,1), T_PCA(i,2), 300, cvec(class(i)), 'filled', 'MarkerEdgeColor','black') 
    hold on
end
set(gca,'linewidth',2,'FontSize',14)
xlabel('PCA Score 1')
ylabel('PCA Score 2')

% Plot the PPA scores
figure
for i=1:length(T_PPA)
   scatter(T_PPA(i,1),T_PPA(i,2),300,cvec(class(i)),'filled','MarkerEdgeColor','black') 
    hold on
end
set(gca,'linewidth',2,'FontSize',14)
xlabel('PP Score 1')
ylabel('PP Score 2')
```
![PCA vs PPA](https://github.com/S-Driscoll/Projection-pursuit/blob/master/common/images/PCA_PPA.PNG)

While PCA reveals 3 clusters corresponding to 2 distinct classes and 2 overlapping classes, PPA is able to reveal 4 distinct clusters corresponding to the 4 different classes.

PPA can also be used to optimize the multivariate kurtosis and the recentered kurtosis. For more information on these options the reader is encouraged to explore the literature linked previously in this repository.
