# Automobile Imports Price Prediction
**Project Team ID:** PTID-CDS-JAN-25-2392  
**Project Code:** PRCP-1017-AutoPricePred  

[GitHub Repository](https://github.com/SURESH745/Auto-Price-Prediction-/tree/main)

---

## Project Overview
In today's competitive automobile market, pricing vehicles accurately is crucial.  
This project builds a machine learning model that predicts car prices based on features like engine size, horsepower, fuel type, and brand reputation — enabling manufacturers, dealerships, and customers to make data-driven decisions.

---

## Domain Analysis

### Industry Overview
The automobile industry is a multi-billion-dollar sector involving manufacturing, sales, and aftermarket services.  
Accurate vehicle pricing is vital for maintaining profitability and market competitiveness.

---

## Feature Overview

### Categorical Features
- Car Manufacturer (e.g., BMW, Audi)
- Fuel Type (Gas/Diesel)
- Aspiration (Standard/Turbo)
- Number of Doors (Two/Four)
- Body Style (Sedan, Hatchback, SUV, Convertible)
- Drive Wheels (FWD, RWD, 4WD)
- Engine Location (Front/Rear)
- Engine Type (DOHC, OHC, Rotor, etc.)
- Fuel System (MPFI, SPFI, IDI)
- Number of Cylinders (Two, Four, Six, etc.)

### Numerical Features
- Symboling (Insurance risk rating)
- Normalized Losses (Adjusted insurance claims)
- Wheelbase, Length, Width, Height
- Curb Weight
- Engine Size, Bore, Stroke
- Compression Ratio
- Horsepower
- Peak RPM
- City MPG, Highway MPG
- Price (Target Variable)

---

## Business Case

### Summary
Pricing varies based on brand image, performance, fuel economy, and market conditions.  
This project aims to predict car prices accurately using machine learning, empowering strategic product positioning and marketing.

### Business Problem
Car prices are influenced by:
- Brand Reputation (e.g., BMW vs. Toyota)
- Engine Power and Fuel Efficiency
- Market Demand & Depreciation
- Geographical Pricing Variations

---

## Data Overview
- Records: 200 car models
- Features: 26 (25 independent + 1 target)
- Key Features: Brand, Fuel Type, Body Style, Engine Size, Horsepower
- Target Variable: Car Price

---

## Project Objectives
- Build an accurate machine learning model to predict car prices.
- Identify key factors affecting vehicle pricing.
- Provide actionable insights for strategic business decisions.

---

## Techniques Used
- Data Cleaning: Handling missing values and correcting data types.
- Feature Engineering: Label Encoding, Feature Scaling, Log Transformation.
- Model Building: Linear Regression, Random Forest, K-Nearest Neighbors (KNN), XGBoost.
- Model Tuning: Hyperparameter tuning using GridSearchCV.
- Evaluation Metrics: R² Score, Adjusted R² Score, MAE, RMSE.

---

## Model Comparison Report

| Model                         | R² Score | Adjusted R² Score |
|--------------------------------|----------|-------------------|
| Random Forest Regressor        | 0.8773   | 0.6216            |
| K-Nearest Neighbors Regressor  | 0.8730   | 0.6084            |
| XGBoost Regressor (Best Model) | 0.8937   | 0.6723            |

**XGBoost** achieved the best performance with the highest R² and Adjusted R² scores.

---

## Final Results
- **Best Model:** XGBoost Regressor
- **Performance:** High accuracy, strong generalization on unseen data
- **Business Use:** Recommended for production deployment in automobile pricing systems

---

## Challenges Faced and Solutions

| Challenge                        | Solution |
|----------------------------------|----------|
| Missing Values ('?')             | Replaced with NaN and imputed using mean/mode |
| Scaling Issues (for KNN)          | Applied StandardScaler |
| Model Overfitting (Decision Trees) | Hyperparameter tuning using GridSearchCV |
| Linear Model Limitations          | Shifted to tree-based models for better fit |

---

## Technologies Used
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib
- **Tools:** Jupyter Notebook
- **Techniques:** Supervised Learning (Regression), Hyperparameter Tuning

---

## Key Visualizations
- Price Distribution (Before and After Log Transformation)
- Correlation Heatmap
- Model Performance Comparison (Bar Chart)
- Feature Importance Analysis

---

## Future Work
- Deploy the XGBoost model using Flask or FastAPI.
- Create an interactive dashboard using Streamlit or Power BI.
- Expand the dataset to include more recent car models and additional features like safety ratings and resale value.

---

## Connect with Me
If you found this project insightful, feel free to star the repository and connect with me on [LinkedIn](https://www.linkedin.com/).  
I am passionate about solving real-world problems with Data Science!

---

## Project Structure
