# Titanic-Survival-Analysis-Demo
This is a simple Python project that demonstrates data processing and machine learning prediction using the Titanic dataset. The project includes two scripts:

survival_analysis.py: Calculates the number of survivors and generates a CSV file of surviving passengers.

survival_model.py: Uses a logistic regression model to predict passenger survival probability and outputs the predictions to a CSV file.

Dependencies

Python 3.10
Packages: pandas, numpy, statsmodels (install using pip install pandas numpy statsmodels )

How to Use

Data Preparation: Ensure the titanic.csv file is in the same directory (containing fields such as PassengerId, Survived, Pclass, Sex, Age, etc.).

Running the script:

Calculate the number of survivors: python survival_analysis.py

Output: surviving_passengers.csv (surviving passengers) and the console displays the total number of survivors.

Model building and prediction: python survival_model.py

Output: titanic_predictions.csv (with prediction fields such as Predicted_Survival_Prob and Predicted_Survival) and model summary.

Example Output

Number of survivors: 342 (based on the standard dataset).

Model Interpretation: Females (Sex=1) and higher cabin class (lower Pclass values) have a higher probability of survival.

Notes

If the data contains NaN values, the script will automatically fill them.

This is a demo project designed to demonstrate Python automation and ML skills. If you need to expand, you can add more features or models.

Feedback is welcome if you have any questions!

Citation
Will Cukierski. Titanic - Machine Learning from Disaster. https://kaggle.com/competitions/titanic, 2012. Kaggle.
