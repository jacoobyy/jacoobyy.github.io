---
layout: post
title: Creating Diagnostic Plots in Python
subtitle: and how to interpret them
gh-repo:
gh-badge:
tags: [OLS, diagnostic plots, python, linear regression]
---

During my time working in market sizing and using general linear models, the group that I was in used R as the main programming language. This decision was made because of how the team was structured and the strengths of each member. R was primarily built as a data analysis tool with an emphasis on statistical features and is great for all sorts of statistical models.

When considering different linear regression models or interpreting if the model you currently built is capturing all of the variance it is useful to look at diagnostic plots.

# What are diagnostic plots?
Let's look at an example of this in R using the Boston housing data.

```r
library(MASS)
Boston
model <- lm(medv ~ ., data=Boston)

par(mfrow=c(2,2))

plot(model)
```

Which plots the following images

![R Plots](../img/rplots.png)

First, we will recreate these plots in Python and then we'll go into interpretation of them towards the end of this blog post.

We'll begin by importing the relevant libraries necessary for building our plots and reading in the data.

**Libraries**
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

%matplotlib inline
```

**Data**
```python
from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target)
```
