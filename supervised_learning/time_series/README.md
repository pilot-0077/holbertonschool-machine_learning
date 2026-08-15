# Time Series

This project focuses on analyzing and modeling data that is ordered over time.
It introduces the main ideas needed to prepare sequential observations and use
them for forecasting and prediction tasks.

## Learning Objectives

By the end of this project, I should be able to explain:

- What a time series is
- How time-dependent data differs from ordinary tabular data
- What trends, seasonality, and noise are
- How to prepare time series data for machine learning
- How sliding windows can be used to create training examples
- How recurrent and deep learning models can be applied to forecasting
- How to evaluate predictions on sequential data

## Core Concepts

Time series models must preserve the temporal order of observations. Data is
usually transformed into input windows and future targets so that a model can
learn relationships between past values and later outcomes. Proper splitting
and evaluation are important because future data must not leak into training.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/time_series`
