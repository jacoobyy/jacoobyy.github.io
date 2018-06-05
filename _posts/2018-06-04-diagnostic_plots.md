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

# generate OLS model
model = sm.OLS(y, sm.add_constant(X))
model_fit = model.fit()
```

First up is the **Residuals vs Fitted** plot. This graph shows if there are any nonlinear patterns in the residuals, and thus in the data as well. An example of this is trying to fit the function $$ f(x) = x^2 $$ with a linear regression $$ y ~ \beta_0 + \beta_1 x $$ . Clearly, the relationship is nonlinear and thus the residuals will look similarly bow-shaped.

**The Code**

```python
# create dataframe from X, y for easier plot handling
dataframe = pd.concat([X, y], axis=1)

# model values
model_fitted_y = model_fit.fittedvalues
# model residuals
model_residuals = model_fit.resid
# normalized residuals
model_norm_residuals = model_fit.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# leverage, from statsmodels internals
model_leverage = model_fit.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = model_fit.get_influence().cooks_distance[0]

plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, dataframe.columns[-1], data=dataframe,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals');
```

which yields the following plot

![Residuals vs Fitted](img/residplot1.png)
