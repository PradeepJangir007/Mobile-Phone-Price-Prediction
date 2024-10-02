# Smart Phone Price Prediction
This project provides a web-based interface for predicting the price of smartphones using a Partial Least Squares Regression (PLSR) model. The model is built using various smartphone features such as RAM, battery capacity, camera specifications, etc. The application is deployed using Streamlit, offering an interactive interface for users to input their smartphone details and predict the price with confidence intervals.

## Table of Contents
### Overview
- Features
- Project Structure
- Modeling and Analysis
- Requirements
- Installation
- Usage
- Model Pipeline
- Contributing

## Overview
This project is a machine learning model designed to predict the price of smartphones based on their features. Users can select from several smartphone attributes, and the system will predict the price with a 90% confidence interval. The backend uses a Partial Least Squares Regression (PLSR) model trained on a dataset of smartphone specifications.

## Core Features:
- Predict smartphone prices based on user inputs.
- Model utilizes PLSR, Polynomial Features, and Box-Cox transformations for enhanced performance.
- Confidence interval of the price prediction.
## Features
### Input Fields:
- Select brand, model series, and features like RAM, processor, dual-sim, display size, battery capacity, etc.
- Specify camera and core counts.
### Price Prediction:
- Predicts smartphone prices with a logarithmic transformation and returns the final price in   a readable format.
- Displays confidence intervals for prediction reliability.
## Project Structure
```markdown
smart_phone_price_prediction/
|-- │
├── data/
│   └── df.pkl                      # Preprocessed dataset file
│   └── smart_phone_price_with_PLSR.pkl  # Trained PLSR model file
│
├── my_module/
│   ├── SparseToDenseTransformer    # Custom module
├── app.py                          # Main Streamlit app
├── README.md                       # This README file
├── requirements.txt                # Python package dependencies
└── notebooks/                      # Jupyter notebooks for EDA and modeling
```
## Modeling and Analysis
### Data Preprocessing:

- Dropped unnecessary columns and handled missing data by filling NaN values with appropriate     values (like mean, mode, or zero for memory features).
- Added derived features such as PPI (Pixels Per Inch).
- Categorical variables (e.g., company, processor) were one-hot encoded.
### Modeling:

- Used Partial Least Squares Regression (PLSR) with a combination of Polynomial Features to     handle non-linearity.
- Applied Box-Cox transformation to standardize the data.
- Built a pipeline to combine preprocessing and modeling steps.
### Evaluation:

- Achieved an R² score of 0.96 on the training set and 0.92 on the test set.
- Calculated confidence intervals for price predictions.
## Requirements
- numpy
- pandas
- scikit-learn
- matplotlib
- streamlit
- pickle
- scipy
- seaborn
## Installation
Clone the repository:
```python
git clone https://github.com/PradeepJangir007/smart-phone-price-prediction.git
cd smart-phone-price-prediction
```
## Run the Streamlit app:
```python
streamlit run app.py
```
## Usage
- Launch the app using Streamlit.
- Fill in the smartphone details such as Brand, Model series, RAM, Battery, Processor, and other features.
- Click on "Predict Price" to get the estimated price.
- The app will also provide a 90% confidence interval for the prediction.
## Model Pipeline
The model pipeline consists of the following steps:

### Feature Transformation:
- Categorical features are one-hot encoded.
- Polynomial features are generated for non-linearity.
- A Box-Cox transformation is applied to normalize the data.
###  Modeling:
- A PLSR model is trained on transformed data.
- The final prediction is made using this model.
## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you have any suggestions or bug reports.

- Fork the repository.
- Create a new branch for your feature (git checkout -b feature-branch).
- Commit your changes (git commit -am 'Add a feature').
- Push to the branch (git push origin feature-branch).
- Create a new pull request.
