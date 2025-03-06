
# customer_purchase_prediction & Micro_Numerosity_Analysis 
<img src="https://github.com/Gtshivanand/customer_purchase_prediction_Micro_Numerosity_Analysis/blob/main/Images/customer's.png"/>
This project aims to forecast customer purchasing behavior by analysing demographic and review-related data, with an emphasis on Micro-Numerosity analysis to evaluate the effects of small-scale numerical factors on purchasing predictions.


## 📊 Project Overview:
The Customer Purchase Prediction & Micro-Numerosity Analysis project aims to forecast customer purchasing behavior by analyzing demographic and review-related data. It places a strong emphasis on Micro-Numerosity analysis to evaluate the effects of small-scale numerical factors on purchasing predictions. This analysis helps businesses improve marketing strategies by understanding and predicting customer purchase intentions more accurately.

## 📄 Abstract:
This project uses a Random Forest Classifier to predict customer purchase behavior by analyzing demographic and review data. It explores how Micro-Numerosity influences prediction accuracy, with model performance evaluated through metrics like *accuracy, **confusion matrix, **ROC curve, and **AUC. Insights from feature importance and model confidence are provided, with future improvements planned through *hyperparameter tuning and feature engineering.

## 📌 Problem Statement:
The project aims to predict customer purchasing behavior using demographic and review-related data, focusing on the impact of small-scale numerical features (Micro-Numerosity) on prediction accuracy.

# 📊Dataset overview:
   
    1.Customer ID: A unique identifier for each customer.
    
    2.Age: The age of the customer.
    
    3.Gender: The gender of the customer.
    
    4.Education: The highest level of education attained by the customer (e.g., School, UG, PG).
    
    5.Review: A categorical rating of the customer’s review (e.g., Average, Poor, Good).
    
    6.Purchased: Whether the customer made a purchase (Yes/No).
    
    This dataset can be used to predict whether a customer will make a purchase based on demographic and review data, with the target variable being the Purchased column. You would need to encode categorical features such as Gender, Education, and Review before applying machine learning algorithms.

## 📂 Project Overview:
This project focuses on predicting customer purchase behavior using machine learning models and analysing the influence of small-scale numerical features through **Micro-Numerosity Analysis**. By leveraging a Random Forest Classifier, we aim to provide insights into what factors drive customer purchases based on demographic and review-related features. This analysis has practical implications for businesses seeking to understand and predict consumer behavior more accurately.

# *Process Overview*:

1. **Data Import & Exploration**
   - Imported relevant libraries and loaded the customer data.
   - Conducted a preliminary exploration to familiarize with data structure and characteristics.

2. **Data Preprocessing**
   - Defined target (`y`) and feature variables (`X`).
   - Encoded categorical variables to prepare for modelling.

3. **Train-Test Split**
   - Divided the dataset into training (80%) and testing (20%) subsets.

4. **Model Training & Selection**
   - Selected a Random Forest Classifier for the prediction task.
   - Trained the model using the training data subset.

5. **Model Evaluation**
   - Predicted outcomes on the test set with the trained model.
   - Assessed model performance through a confusion matrix, accuracy score, and classification report.

6. **Feature Importance Analysis**
   - Visualized feature importance using a bar chart.
   - Identified the primary factors affecting purchase predictions.

7. **Confusion Matrix Heatmap**
   - Generated a heatmap of the confusion matrix to illustrate model performance.
   - Analysed instances of true/false positives and negatives.

8. **ROC Curve & AUC**
   - Computed and plotted the ROC curve and AUC to evaluate model discriminatory ability.
   - Examined the balance between true positive and false positive rates.

9. **Probability Distribution Analysis**
   - Reviewed the distribution of probabilities for the positive ('Yes') class predictions.
   - Assessed model confidence in its purchase predictions.



# 📌 Key Dataset Features:
- **Customer Demographics**: Information about the customer’s age, gender, location, etc.

- **Customer Reviews**: Feedback or ratings given by customers.

- **Purchase Indicator**: Whether a customer made a purchase (target variable).


## 🔍 Project Workflow:

- ###  Data Import and Exploration
  - **Imported necessary libraries** such as Pandas, NumPy, and Matplotlib.
  - **Loaded and explored the dataset** to understand the key variables and identify any missing or inconsistent data.
  - Performed **data visualization** to gain initial insights into customer characteristics and purchase behavior.

- ### Data Preprocessing
  - **Target Variable (y)**: Defined the purchase status as the target for prediction.
  - **Feature Variables (X)**: Selected relevant demographic and review-based features for model training.
  - **Categorical Encoding**: Applied label encoding or one-hot encoding to transform categorical features for model compatibility.
  - **Data Normalization**: Scaled features to ensure uniformity across all variables.

- ###  Train-Test Split
  - **80-20 split**: Divided the data into 80% for training the model and 20% for testing its performance.
  - Ensured that both sets maintained a balanced distribution of the target variable.

- ###  Model Selection and Training
  - **Random Forest Classifier**: Selected as the machine learning model for its robustness and ability to handle feature importance.
  - Trained the model on the training dataset, adjusting hyperparameters where necessary to optimize performance.

- ###  Model Evaluation
  - **Accuracy Metrics**: Evaluated the model using standard metrics like the **accuracy score**, **confusion matrix**, and **classification report**.
  - **Confusion Matrix Analysis**: Illustrated true positives, false positives, true negatives, and false negatives using a heatmap to visualize model performance across different prediction categories.

- ###  Feature Importance Visualization
  - Plotted a **bar chart** to visualize the importance of each feature in predicting purchases.
  - Identified key factors that have a significant impact on customer purchase decisions, such as demographic attributes and review scores.

- ###  ROC Curve and AUC Score
  - Generated the **ROC curve** to evaluate the trade-offs between sensitivity (True Positive Rate) and specificity (False Positive Rate).
  - Calculated the **AUC (Area Under the Curve)** score to quantify the model’s ability to distinguish between classes.

- ###  Distribution of Predicted Probabilities
  - Analysed the **distribution of predicted probabilities** for positive class (customers predicted to make a purchase).
  - This analysis helped us understand the model’s confidence in its predictions and identify areas for potential improvement.

---

##  📈 Key Takeaways:
- **Model Performance**: Achieved an accuracy of **60%** on the test set, indicating room for further optimization.
- **Feature Importance**: The analysis revealed that customer demographics, particularly age and review feedback, were strong predictors of purchase behavior.
- **Confusion Matrix**: Provided insights into the model’s classification performance, identifying misclassified instances for further analysis.
- **ROC and AUC Analysis**: The AUC score indicated moderate discrimination power, with the ROC curve providing a visual assessment of model performance.
- **Probability Distribution**: Highlighted how confident the model was in predicting purchases, offering insights into the strength of its predictions.

---

## 📌 Next Steps:
  
**Model Improvement**: Further refine the model by conducting 
**hyperparameter tuning** :to improve accuracy and overall performance.
**Feature Engineering**: Investigate the potential for **additional features** or feature transformations to enhance the model's predictive power.
**Model Comparisons**: Test alternative machine learning algorithms such as Gradient Boosting or XGBoost to explore better-performing models.
**Micro-Numerosity Deep Dive**: Explore more advanced techniques to understand the specific effects of small-scale numerical features on customer behavior.


## 🛠 Technologies Used:
**Python**: Programming language used for data manipulation, analysis, and modelling.
**Pandas & NumPy**: For data handling and preprocessing.
**Matplotlib & Seaborn**: For data visualization.
**Scikit-learn**: Machine learning library used to build the predictive model.
**Jupyter Notebook**: Development environment for code implementation and analysis.


# 📈 Conclusion:
This project demonstrates the use of machine learning techniques to predict customer purchase behavior while incorporating Micro-Numerosity Analysis to gain deeper insights into the significance of small-scale numerical features. By using a Random Forest Classifier, we achieved an initial accuracy of 60%, with opportunities for further enhancement through model optimization and feature engineering.

The analysis highlights the importance of understanding subtle factors that can drive consumer purchasing decisions, providing actionable insights for businesses aiming to improve marketing strategies and customer targeting.

## 📧  Feedback and Suggestions:

Thank you for visiting my repository! If you have any questions or feedback, feel free to reach out.

I’d love to hear your thoughts, feedback, and suggestions! Feel free to connect with me:

 LinkedIn: [Shivanand Nashi](https://www.linkedin.com/in/shivanand-s-nashi-79579821a)
 
 Email: shivanandnashi97@gmail.com


Looking forward to connecting and exchanging ideas!

## ✨ Support this project!
If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!
Your support helps keep the project active and encourages further development.

Thank you for your support! 💖

