import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Feature Engineering : Calculate missing values in city column

def missing_city_count():
    """finding the number of missing values in the city column."""
    print("="*50)
    print("Missing City Values")
    print("="*50)
    
    # Read CSV
    data = pd.read_csv("house_sales.csv")
    
    # Checking for missing values in city column (replace '--' with NaN)
    data['city'].replace('--', pd.NA, inplace=True)
    missing_city = data['city'].isna().sum()
    
    print(f"Number of missing values in city column: {missing_city}")
    return missing_city


# Feature Engineering and preprocessing: Data Cleaning

def task2_data_cleaning():
    print("="*50)
    print("Data Cleaning")
    print("="*50)
    
    # Read data from CSV
    data = pd.read_csv("house_sales.csv")
    
    # Convert house_id column to nominal
    data['house_id'] = data['house_id'].astype('category')
    
    # Handle 'city' column: Replace '--' and missing values with "Unknown"
    data['city'] = data['city'].replace('--', 'Unknown')
    data['city'] = data['city'].fillna('Unknown')
    
    # Filtering only valid cities
    valid_cities = ['Silvertown', 'Riverford', 'Teasdale', 'Poppleton', 'Unknown']
    data = data[data['city'].isin(valid_cities)]
    data['city'] = data['city'].astype('category')
    
    # Handle 'sale_price' column: Remove missing entries 
    data = data.dropna(subset=['sale_price'])
    data = data[data['sale_price'] >= 0]
    
    # Replace missing values with "2023-01-01"
    data['sale_date'] = pd.to_datetime(data['sale_date'], errors='coerce')
    data['sale_date'] = data['sale_date'].fillna(pd.Timestamp("2023-01-01"))
    data['sale_date'] = data['sale_date'].dt.strftime('%Y-%m-%d')
    data['sale_date'] = data['sale_date'].astype('category')
    
    # Imputation: Replace 'NA' with NaN and fill with mean
    data['months_listed'] = data['months_listed'].replace('NA', pd.NA)
    mean_months_listed = round(data['months_listed'].mean(), 1)
    data['months_listed'] = data['months_listed'].fillna(mean_months_listed)
    
    # Imputation: Replace missing values with mean rounded to nearest integer
    mean_bedrooms = round(data['bedrooms'].mean())
    data['bedrooms'] = data['bedrooms'].fillna(mean_bedrooms)
    data = data[data['bedrooms'] >= 0]
    
    # Replacing abbreviations and fill missing values
    data['house_type'] = data['house_type'].replace({
        'Det.': 'Detached',
        'Semi': 'Semi-detached',
        'Terr.': 'Terraced'
    })
    
    # Convert to ordinal categorical data
    category_order = ['Terraced', 'Semi-detached', 'Detached']
    data['house_type'] = pd.Categorical(data['house_type'], categories=category_order, ordered=True)
    
    # Fill missing house types with most common
    most_common_type = data['house_type'].mode()[0]
    data['house_type'] = data['house_type'].fillna(most_common_type)
    
    # converting area column to numeric
    data['area'] = data['area'].astype(str).str.replace('sq.m.', '').str.strip()
    data['area'] = pd.to_numeric(data['area'], errors='coerce')
    mean_area = round(data['area'].mean(), 1)
    data['area'] = data['area'].fillna(mean_area)
    
    clean_data = data
    print("Data cleaning completed!")
    print(f"Cleaned dataset shape: {clean_data.shape}")
    print("\nFirst 5 rows:")
    print(clean_data.head())
    
    return clean_data

# Price Analysis

def price_by_bedrooms():
    """Analyze average sale price and variance by number of bedrooms."""
    print("="*50)
    print("TASK 3: Price Analysis by Bedrooms")
    print("="*50)
    
    data = task2_data_cleaning()
    
    # Conversion
    data['sale_price'] = pd.to_numeric(data['sale_price'], errors='coerce')
    
    # Grouping by bedrooms and calculate average and variance of sale_price
    price_by_rooms = data.groupby('bedrooms')['sale_price'].agg(['mean', 'var']).reset_index()
    price_by_rooms = price_by_rooms.rename(columns={'mean': 'avg_price', 'var': 'var_price'})
    
    price_by_rooms['avg_price'] = price_by_rooms['avg_price'].round(1)
    price_by_rooms['var_price'] = price_by_rooms['var_price'].round(1)
    
    print("Price analysis by number of bedrooms:")
    print(price_by_rooms)
    
    return price_by_rooms


# Applying Ordinary Least Square model

def preprocess_data(data, is_training=True):
    """Preprocess data for model training/prediction."""
    
    # Handle missing values
    data['city'] = data['city'].fillna('Unknown')
    
    if is_training:
        
        data = data.dropna(subset=['sale_price'])
        data = data[data['sale_price'] >= 0]
    
    data['sale_date'] = data['sale_date'].fillna('2023-01-01')
    
    if 'months_listed' in data.columns:
        mean_months = round(data['months_listed'].mean(), 1)
        data['months_listed'] = data['months_listed'].fillna(mean_months)
    
    if 'bedrooms' in data.columns:
        mean_bedrooms = round(data['bedrooms'].mean())
        data['bedrooms'] = data['bedrooms'].fillna(mean_bedrooms)
    
    if 'house_type' in data.columns:
        mode_house_type = data['house_type'].mode()[0] if not data['house_type'].mode().empty else 'Detached'
        data['house_type'] = data['house_type'].fillna(mode_house_type)
    
    if 'area' in data.columns:
        mean_area = round(data['area'].mean(), 1)
        data['area'] = data['area'].fillna(mean_area)
    
    return data


def baseline_model():
    """Fit OLS"""
    print("="*50)
    print("Ordinary Least Square")
    print("="*50)
    
    try:
        # train and validate
        train_data = pd.read_csv("train.csv")
        validation_data = pd.read_csv("validation.csv")
        
        # Preprocess data
        train_data = preprocess_data(train_data, is_training=True)
        validation_data = preprocess_data(validation_data, is_training=False)
        
        # Encoding categorical variables
        train_data = pd.get_dummies(train_data, columns=['house_type', 'city'], drop_first=True)
        validation_data = pd.get_dummies(validation_data, columns=['house_type', 'city'], drop_first=True)
        
        # Handle date features
        train_data['sale_date'] = pd.to_datetime(train_data['sale_date'])
        validation_data['sale_date'] = pd.to_datetime(validation_data['sale_date'])
        
        # Extract date features
        for data in [train_data, validation_data]:
            data['sale_year'] = data['sale_date'].dt.year
            data['sale_month'] = data['sale_date'].dt.month
            data['sale_day'] = data['sale_date'].dt.day
            data.drop('sale_date', axis=1, inplace=True)
        
        # Align columns between train and validation datasets
        train_cols = set(train_data.columns) - {'sale_price'}
        val_cols = set(validation_data.columns) - {'house_id'}
        
        # Add missing columns with zeros
        for col in train_cols - val_cols:
            validation_data[col] = 0
        for col in val_cols - train_cols:
            train_data[col] = 0
        
        # Ensure same column order
        feature_cols = sorted(list(train_cols & val_cols))
        
        # Fit the model
        model = LinearRegression()
        X_train = train_data[feature_cols]
        y_train = train_data['sale_price']
        
        model.fit(X_train, y_train)
        
        # Make predictions
        X_val = validation_data[feature_cols]
        predicted_prices = model.predict(X_val)
        
        # Create results dataframe
        base_result = pd.DataFrame({
            'house_id': validation_data['house_id'], 
            'price': predicted_prices
        })
        
        print("Baseline model training completed!")
        print("First 5 predictions:")
        print(base_result.head())
        
        return base_result
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Required files: train.csv and validation.csv not found")
        return None


# Random Forest Model
def comparison_model():
    """Fit a Random Forest Regressor model for comparison."""
    print("="*50)
    print("TASK 5: Random Forest Comparison Model")
    print("="*50)
    
    try:
        # Load the training and validation data
        train_data = pd.read_csv("train.csv")
        validation_data = pd.read_csv("validation.csv")
        
        # Preprocess data (same as baseline model)
        train_data = preprocess_data(train_data, is_training=True)
        validation_data = preprocess_data(validation_data, is_training=False)
        
        # Encoding categorical variables
        train_data = pd.get_dummies(train_data, columns=['house_type', 'city'], drop_first=True)
        validation_data = pd.get_dummies(validation_data, columns=['house_type', 'city'], drop_first=True)
        
        # Handle date features
        train_data['sale_date'] = pd.to_datetime(train_data['sale_date'])
        validation_data['sale_date'] = pd.to_datetime(validation_data['sale_date'])
        
        # Extract date features
        for data in [train_data, validation_data]:
            data['sale_year'] = data['sale_date'].dt.year
            data['sale_month'] = data['sale_date'].dt.month
            data['sale_day'] = data['sale_date'].dt.day
            data.drop('sale_date', axis=1, inplace=True)
        
        # Align columns between train and validation datasets
        train_cols = set(train_data.columns) - {'sale_price'}
        val_cols = set(validation_data.columns) - {'house_id'}
        
        # Add missing columns with zeros
        for col in train_cols - val_cols:
            validation_data[col] = 0
        for col in val_cols - train_cols:
            train_data[col] = 0
        
        # Ensure same column order
        feature_cols = sorted(list(train_cols & val_cols))
        
        # Fit the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train = train_data[feature_cols]
        y_train = train_data['sale_price']
        
        model.fit(X_train, y_train)
        
        # Make predictions
        X_val = validation_data[feature_cols]
        predicted_prices = model.predict(X_val)
        
        # Create results dataframe
        compare_result = pd.DataFrame({
            'house_id': validation_data['house_id'], 
            'price': predicted_prices
        })
        
        print("Random Forest model training completed!")
        print("First 5 predictions:")
        print(compare_result.head())
        
        return compare_result
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Required files: train.csv and validation.csv not found")
        return None



# MAIN EXECUTION

def main():
    """Execute all tasks in sequence."""
    print("="*70)
    
    # Execute all tasks
    try:
    
        missing_city = missing_city_count()
        
        clean_data = task2_data_cleaning()
        
        price_by_rooms = price_by_bedrooms()
        
        base_result = baseline_model()
        
        compare_result = comparison_model()
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("- house_sales.csv")
        print("- train.csv") 
        print("- validation.csv")


if __name__ == "__main__":
    main()