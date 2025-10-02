# machinelearning_tasks

# Data Science & Machine Learning Tasks

This repository contains solutions to a series of **Data Science** and **Machine Learning** tasks, covering topics such as Exploratory Data Analysis (EDA), Predictive Modeling, Product Recommendation Systems, and Time Series Forecasting. These tasks are aimed at building practical skills in handling data, creating machine learning models, and evaluating their performance.


## Task Overview

### **Task 1: Understanding the Customer's Story (EDA)**
This task involves performing **Exploratory Data Analysis (EDA)** on a retail sales dataset. The goal is to uncover patterns in customer purchases, peak sales times, and product categories. The EDA results are visualized using **Matplotlib** and **Seaborn**.

### **Task 2: Visualizing for Impact**
In this task, an **interactive dashboard** is created using **Plotly** to visualize key findings from the EDA, including top-selling products, sales by country, and peak shopping hours. The dashboard allows for filtering data based on product categories and countries.

### **Task 3: Predicting Customer Churn (Classification)**
This task involves building a **Logistic Regression** model to predict customer churn. The dataset is preprocessed by encoding categorical features and splitting it into training and test sets. Model performance is evaluated using accuracy and confusion matrix.

### **Task 4: Optimizing Churn Prediction (Advanced Classification)**
Here, we enhance the churn prediction model from Task 3 by using a **Random Forest** classifier. We apply feature scaling and evaluate the model's performance using **Precision**, **Recall**, and **F1-Score**. We compare the results with the Logistic Regression model to identify the best-performing model.

### **Task 5: Building a Product Recommendation System**
In this task, we build a **content-based recommendation system** using the **TMDB 5000 Movie Dataset**. The system uses **TF-IDF Vectorizer** and **Cosine Similarity** to recommend movies based on their descriptions. The system returns the top 5 most similar movies for a given movie.

### **Task 6: Forecasting Future Sales (Time Series Analysis)**
This task involves building a **SARIMA model** to forecast sales for the next 30 days based on historical sales data. The dataset is checked for stationarity, and if necessary, differenced to make it stationary. The model's predictions are visualized alongside historical data.

Expected Outcomes

Task 1: Visualizations showing the distribution of customer sales, sales by product category, and sales over time.

Task 2: An interactive dashboard that displays top-selling products, sales by country, and peak shopping hours.

Task 3: A churn prediction model using Logistic Regression with an accuracy score and confusion matrix.

Task 4: An improved churn prediction model using Random Forest with performance metrics (Precision, Recall, F1-Score).

Task 5: A content-based movie recommendation system that returns the most similar movies based on a given movie's description.

Task 6: A SARIMA model that forecasts future sales for the next 30 days and visualizes the forecast alongside historical sales.

## Prerequisites

To run this project, you need to have **Python 3** installed. You will also need to install the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- plotly

You can install the required libraries using **pip**:

