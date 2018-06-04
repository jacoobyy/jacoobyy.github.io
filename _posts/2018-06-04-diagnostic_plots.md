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
