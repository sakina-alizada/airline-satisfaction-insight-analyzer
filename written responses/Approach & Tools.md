# Approach & Tool Selection

To address the problem of analyzing airline passenger satisfaction data, I designed a data analytics pipeline that combines data processing, machine learning, and generative AI.

The first stage of the project focuses on data processing. I used Python with the pandas and numpy libraries to load, clean, and manipulate the dataset. This included standardizing column names, handling missing values, identifying categorical and numeric variables, and preparing the data for machine learning models. Exploratory data analysis was performed using matplotlib and seaborn to visualize patterns in passenger satisfaction and better understand the relationships between service ratings and the target variable.

The second stage involves machine learning. I implemented two classification models using scikit‑learn: Logistic Regression and Random Forest. Logistic Regression was selected as a baseline model because it is simple, interpretable, and commonly used for binary classification tasks. Random Forest was chosen as the primary model because it can capture complex, non‑linear relationships between features and typically performs well on structured datasets like surveys. Both models were evaluated using metrics such as accuracy, precision, recall, F1 score, and confusion matrix.

The final stage of the project demonstrates the use of generative AI. After the models are trained and evaluated, a module generates a natural‑language insight report summarizing key findings such as satisfaction distribution, important predictors of satisfaction, and model performance. This step illustrates how generative AI can help translate technical analytics results into insights that are easier for non‑technical stakeholders to understand.

Alternative approaches could have included using more complex algorithms such as gradient boosting or neural networks. However, I selected Random Forest and Logistic Regression because they offer a strong balance between performance, interpretability, and implementation complexity for a capstone project.
