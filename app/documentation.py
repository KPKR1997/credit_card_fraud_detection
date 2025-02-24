import streamlit as st

st.title("Credit card fraud detection")
st.caption('Author: Krishnaprakash | 02/08/2024')
st.divider()
with st.container():
    st.write('''
    Introduction: The Growing Need for Fraud Detection in the Digital Age
    With rapid advancements in technology, online transactions have become an essential part of daily life. From e-commerce purchases to digital banking, people rely on credit and debit cards more than ever. However, with this convenience comes a rising threat—credit card fraud.

    Financial fraud has become increasingly sophisticated, with cybercriminals using advanced tactics to exploit security loopholes. Traditional fraud detection systems, which relied on rule-based methods and manual reviews, struggled to keep up with evolving fraud patterns. These systems often led to false positives, delayed transactions, and an inability to detect emerging fraud techniques.
    ''')
    st.subheader('The Need for Advanced Fraud Detection')
    st.write('''
    To combat this growing challenge, banks and financial institutions must adopt intelligent fraud detection mechanisms. Machine learning (ML) and artificial intelligence (AI) have revolutionized the way fraud is detected by analyzing vast amounts of transactional data in real time. Unlike traditional methods, ML-based models can: \n
    1.Detect hidden fraud patterns that rule-based systems miss.\n
    2.Reduce false alarms by learning from past legitimate transactions.\n
    3.Continuously adapt to new fraud techniques using real-time data.\n
    4.Improve accuracy by leveraging multiple transaction features, such as location, time, amount, and user behavior.\n    
    ''')
   

    st.subheader('How Machine Learning Helps Prevent Credit Card Fraud')
    st.write('''
    Modern fraud detection systems use ML algorithms to analyze transaction data and classify transactions as fraudulent or legitimate. These models learn from past fraudulent cases and can make predictions based on patterns such as: \n
    🔹 Unusual spending behavior (e.g., a sudden large transaction in another country). \n
    🔹 Transactions from a suspicious location or device. \n
    🔹 Repeated failed login attempts or high-frequency transactions in a short time. \n

    Financial institutions now integrate real-time fraud detection models that instantly flag suspicious transactions. This prevents unauthorized access, protects cardholders, and helps banks reduce losses due to fraudulent activities.
    ''')
    st.subheader('Building a Fraud Detection Model')
    st.write('''In this project, we aim to develop a machine learning-based credit card fraud detection system. Our model will analyze transaction data, identify fraud patterns, and provide real-time predictions to help prevent fraudulent transactions. The repository is structured to support data preprocessing, model training, evaluation, and deployment in a user-friendly application.
    ''')

    st.header('Abstract')
    
    st.write('''
        As the number of online transactions is on the rise, credit card fraud is becoming a serious issue for both consumers and financial institutions. Rule-based fraud detection systems are inefficient and produce false positives since they are unable to cope with changing fraud patterns. In order to solve this issue, this project uses machine learning (ML) methods to build an effective fraud detection model.

        The data set includes 99,977 transactions that are characterized by 16 features, such as transaction information such as time, amount, card type, entry mode, merchant category, and user information. The target variable, "Fraud," shows whether a transaction is fraudulent (1) or legitimate (0). The model does data preprocessing by using Standard Scaling on numerical attributes and One-Hot Encoding on categorical variables.

        In the classification, ensemble learning is employed, where multiple models are experimented with GridSearchCV to determine the best algorithm to identify fraudulent transactions. The resulting model is deployed as a real-time fraud detection system in Streamlit, where users can input transaction data and receive real-time predictions of possible fraud.

        This project demonstrates how fraud detection based on ML can strengthen security by identifying true suspicious transactions and minimizing false positives, hence being a valuable tool in anti-financial fraud prevention.
            ''')

    st.subheader('Literature Survey')
    st.write('''A machine learning model is a program that can find patterns or make decisions from a previously unseen dataset. For example, in natural language processing, machine learning models can parse and correctly recognize the intent behind previously unheard sentences or combinations of words. In image recognition, a machine learning model can be taught to recognize objects - such as cars or dogs. A machine learning model can perform such tasks by having it 'trained' with a large dataset. During training, the machine learning algorithm is optimized to find certain patterns or outputs from the dataset, depending on the task. The output of this process 
             - often a computer program with specific rules and data structures - is called a machine learning model.''')
    
    st.write('''
        A machine learning algorithm is a mathematical method to find patterns in a set of data. Machine Learning algorithms are often drawn from statistics, calculus, and linear algebra. Some popular examples of machine learning algorithms include linear regression, decision trees, random forest, and XGBoost.
    ''')
    st.caption('Algorithms used here for model training:')
    st.write('''
        ***Random Forest***: Random forest is a collection of many decision trees from random subsets of the data, resulting in a combination of trees that may be more accurate in prediction than a single decision tree.
    ''')

    st.write('''
        ***Decision Trees***: Decision trees are also classifiers that are used to determine what category an input falls into by traversing the leaf's and nodes of a tree''')

    st.write('''
        ***Logistic Regression***: Logistic Regression is used to determine if an input belongs to a certain group or not''')

    st.write('''
        ***Boosting algorithms***: Boosting algorithms, such as Gradient Boosting Machine, XGBoost, and LightGBM, use ensemble learning. 
        They combine the predictions from multiple algorithms (such as decision trees) while taking into account the error from the previous algorithm.''')

    st.subheader('Structure and Methodology')
    st.write('''The credit card fraud detection project follows a structured methodology to develop a machine learning-based system capable of identifying fraudulent transactions. The project is built using Python, machine learning libraries (Scikit-learn, XGBoost, etc.), and Streamlit for deployment. It involves several key steps, including data preprocessing, model selection, training, and deployment in a web application.

        \nThe dataset contains 99,977 transactions with 16 features, including transaction details such as day of the week, time, card type, entry mode, transaction amount, merchant group, country of transaction, and user demographics like age, gender, and residence. The target variable, "Fraud", indicates whether a transaction is fraudulent (1) or legitimate (0). The dataset also includes Transaction ID and Date, which are excluded from model training as they do not contribute to fraud detection. The dataset is stored in the data/ directory in CSV format, which is later ingested and transformed for training.

        \nThe repository is structured into multiple folders. The logs/ directory stores logs generated by the logging module for debugging and monitoring. The notebook/ directory contains the EDA.ipynb file, which performs Exploratory Data Analysis (EDA) to understand transaction distributions, fraud trends, and correlations. The src/ directory contains core ML pipeline components, error handling modules, and utility functions. The components/ subfolder includes scripts for data ingestion, validation, transformation, model training, and pipeline execution. The data_transformation.py script standardizes numerical features (amount, time, age) using StandardScaler and encodes categorical features (card type, merchant, transaction type, etc.) using One-Hot Encoding. The model_training.py script employs GridSearchCV to optimize model selection and trains an ensemble model (e.g., Random Forest, XGBoost) for fraud detection. The best-performing model is saved in the artifacts/ directory, which stores train/test splits, preprocessing objects, and the trained model (model.pkl).

        \nError handling is managed using logger.py for logging system activities and exception.py for custom exception handling. The utils/ folder contains utility functions like loading and saving models. The machine learning pipeline is orchestrated using pipeline.py, which integrates data ingestion, transformation, training, and prediction. Once trained, the model is deployed using Streamlit, allowing users to input transaction details and receive real-time fraud predictions via the streamlit_app.py script.

        \nThe final fraud detection system is designed to be scalable, accurate, and efficient, leveraging advanced ML techniques to detect anomalies in financial transactions. By integrating machine learning, feature engineering, and hyperparameter optimization, the system enhances fraud prevention strategies, providing a valuable solution for financial institutions.''')
    

st.image("app/flow.jpg", caption="Image: process pipeline flow", width=700)

