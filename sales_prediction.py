# Import the necessary tools for our data work.
# pandas is for handling our data tables.
# seaborn and matplotlib are for creating our charts.
# scikit-learn is for building our prediction model.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def analyze_and_predict_sales():
    """
    This function loads the advertising data, visualizes it,
    and builds a model to predict sales based on TV ad spend.
    """
    print("--- Starting Sales Prediction Analysis ---")

    # Step 1: Load and clean the data
    df = pd.read_csv('Advertising.csv')
    df = df.drop(columns=['Unnamed: 0']) # Remove the extra index column

    # Step 2: Create the visualization
    print("Creating a chart to see how advertising affects sales...")
    sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
    plt.savefig('sales_vs_advertising_scatter.png')
    print("Chart saved as 'sales_vs_advertising_scatter.png'")
    plt.close()

    # --- Step 3: Build a Prediction Model ---
    print("\nBuilding a model to predict sales from TV ad spend...")

    # Define our features (X) and target (y)
    # We are using only 'TV' for this simple model because it has the strongest correlation.
    X = df[['TV']]
    y = df['Sales']

    # Split the data into training and testing sets
    # We'll train the model on 80% of the data and test it on the remaining 20%.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model
    # Let's see how well our model performs on the test data it has never seen before.
    predictions = model.predict(X_test)
    r2_score = metrics.r2_score(y_test, predictions)

    print(f"\nModel Performance:")
    print(f"The R-squared value is: {r2_score:.2f}")
    print("This means our model can explain about 68% of the variation in sales using just TV ad spend.")


# This is the main part of our script that runs the function.
if __name__ == "__main__":
    analyze_and_predict_sales()
    print("\nAnalysis complete.")
