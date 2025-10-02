# train.py (for DecisionTreeRegressor)

from sklearn.tree import DecisionTreeRegressor
from misc import load_boston_data, preprocess_data, train_and_evaluate_model

def main():
    # 1. Data Loading
    df = load_boston_data()
    
    # 2. Data Preprocessing (Split)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 3. Model Training and Evaluation
    model_class = DecisionTreeRegressor
    model_params = {'random_state': 42}
    
    pipeline, mse = train_and_evaluate_model(
        model_class, model_params, X_train, X_test, y_train, y_test
    )
    
    # 4. Display Results
    print("--- Decision Tree Regressor Performance ---")
    print(f"Average Mean Squared Error (MSE) on Test Set: {mse:.4f}")

if __name__ == "__main__":
    main()