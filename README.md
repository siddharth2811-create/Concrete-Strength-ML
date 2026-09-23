# Machine Learning-Based Prediction of Concrete Compressive Strength

## Overview

This project applies machine learning regression techniques to predict the compressive strength of concrete based on its mix composition and curing age.

The project uses the Concrete Compressive Strength dataset from the UCI Machine Learning Repository.

## Objective

To develop and compare machine learning models for predicting concrete compressive strength from material composition and age.

## Dataset

The dataset contains 1,030 concrete samples with eight input features:

- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Fine Aggregate
- Age

Target variable:

- Concrete Compressive Strength (MPa)

## Models Used

Three regression models were implemented:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

## Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Analysis

The project compares model performance and investigates the relative importance of different concrete mix parameters using Random Forest feature importance.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- UCI Machine Learning Repository

## Dataset Source

Yeh, I. (1998). Concrete Compressive Strength.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5PK67
