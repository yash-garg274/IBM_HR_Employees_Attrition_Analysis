# **Employee Attrition Analysis with Machine Learning**

This repository contains a Jupyter Notebook that analyzes employee attrition using the IBM HR Analytics dataset. The project identifies factors contributing to attrition, predicts employee attrition, and visualises key insights.



## **Objective**

The goal of this project is to:
1. Understand the impact of various work-life balance factors on employee attrition.
2. Implement Logistic Regression to find the coefficients of various Work-life balance features and train a machine learning model, Random Forest, to predict attrition.
3. Visualize key metrics and insights for interpretability and decision-making.



## **Dataset**

The dataset used in this project is the **IBM HR Analytics Employee Attrition & Performance** dataset, downloaded from [Kaggle](https://www.kaggle.com/datasets).

### Key Features:
- **Attrition**: Indicates whether an employee has left the organization (`Yes` or `No`).
- **OverTime**: Whether the employee works overtime (`Yes` or `No`).
- **BusinessTravel**: Frequency of business travel (`Travel_Rarely`, `Travel_Frequently`, `Non-Travel`).
- **DistanceFromHome**: Distance between the employee's home and workplace.
- **WorkLifeBalance**: Employee's perception of their work-life balance (scale of 1 to 4).
- **JobSatisfaction**: Job satisfaction level (scale of 1 to 4).
- **EnvironmentSatisfaction**: Work environment satisfaction level (scale of 1 to 4).


## **Workflow**

### 1. Data Preprocessing:
   - Convert categorical variables (e.g., `Attrition`, `OverTime`, `BusinessTravel`) into numerical values for analysis.
   - Scale numerical features using `StandardScaler` to ensure consistent input ranges for machine learning models.

### 2. Model Training:
   - **Logistic Regression**:
     - Provides insight into feature importance through coefficients.
   - **Random Forest Classifier**:
     - Offers robust, high-accuracy predictions by leveraging multiple decision trees.

### 3. Evaluation Metrics:
   - **Classification Report**: Includes precision, recall, F1-score, and accuracy.
   - **ROC-AUC Score**: Assesses the model's ability to distinguish between classes.

### 4. Visualizations:
   - **Impact of Work-Life Balance Factors on Attrition**:
     A bar plot based on Logistic Regression coefficients, highlighting the influence of each feature.
   - **ROC Curve**:
     A Receiver Operating Characteristic curve for the Random Forest model, illustrating its predictive performance.


## **Results**

### Insights:
- Logistic Regression analysis revealed that **OverTime**, **BusinessTravel**, and **DistanceFromHome** were key contributors to employee attrition due to their positive coefficients. 
- This suggests that employees who work overtime, frequently travel for business, or live far from the workplace may feel more burdened, leading to higher attrition rates.
- In the future, the company can address these factors by offering flexible schedules, minimizing unnecessary travel, and supporting relocation options to help employees feel less overburdened, ultimately reducing attrition.

### Key Visualizations:
1. **Feature Coefficients**:
   - A bar plot illustrating the relative importance of work-life balance factors.
2. **ROC Curve**:
   - A curve showcasing the Random Forest model's performance in separating classes.

### Model Performance:
- **Random Forest Classifier** achieved high predictive accuracy for Non-Attrition.


## **How to Use**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/employee-attrition-analysis.git
   cd employee-attrition-analysis
   ```

2. **Open the Jupyter Notebook**:
   - Launch the notebook in your preferred environment (e.g., JupyterLab, Jupyter Notebook, or VS Code).

3. **Place the Dataset**:
   - Add the `IBM_HR_Employees.csv` dataset (downloaded from Kaggle) to the same directory as the notebook.

4. **Run the Notebook**:
   - Execute the cells in sequence to preprocess the data, train models, and visualize results.


## **File Structure**

- **`employee_attrition_analysis.ipynb`**: Jupyter Notebook containing the entire analysis, from preprocessing to model evaluation and visualization.
- **Dataset**: The `IBM_HR_Employees.csv` file needs to be added to the repository for the notebook to run.


## **Future Enhancements**

- Add support for additional machine learning models like Gradient Boosting or XGBoost.
- Explore hyperparameter tuning to optimize model performance.
- Integrate interactive visualizations using tools like Plotly or Dash.


## **Acknowledgments**

- Dataset source: [Kaggle - IBM HR Analytics Employee Attrition & Performance]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)).
- Libraries used: `pandas`, `seaborn`, `matplotlib`, `scikit-learn`.

