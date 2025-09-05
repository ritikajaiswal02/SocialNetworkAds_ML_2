# 🧑‍🤝‍🧑 Social Network Ads – Machine Learning Project  

## 📖 Project Overview  
This project predicts whether a user will **purchase a product** based on their **Age** and **Estimated Salary** using classification models.  
The final chosen model is a **Random Forest Classifier**, known for its robustness and high accuracy.  

The model is trained, evaluated, and then saved as `model.pkl`, making it ready for deployment in a web application (Flask/Django/Streamlit).  

---

## 📂 Project Structure 
```
socialNetworkAds_ML_2/
│
├── s1_visualization/               # Step 1: Data Visualization
│   └── s1_visualization.py         # Visualizing dataset distribution & decision boundaries
│
├── s2_LogisticRvisualization/      # Step 2: Logistic Regression Visualization
│   └── s2_LogisticRvisualization.py# Plotting decision boundary for Logistic Regression
│
├── s3_logisticR/                   # Step 3: Logistic Regression Model
│   └── s3_logisticR.py             # Training & evaluating Logistic Regression classifier
│
├── s4_BernoulliNB/                 # Step 4: Bernoulli Naïve Bayes
│   └── s4_BernoulliNB.py           # Model training & accuracy evaluation
│
├── s5_GaussianNB/                  # Step 5: Gaussian Naïve Bayes
│   └── s5_GaussianNB.py            # Model training & evaluation on continuous features
│
├── s6_KNNC/                        # Step 6: K-Nearest Neighbors Classifier
│   └── s6_KNNC.py                  # Model training & decision boundary visualization
│
├── s7_DTC/                         # Step 7: Decision Tree Classifier
│   └── s7_DTC.py                   # Training & evaluating Decision Tree model
│
├── s8_RFC/                         # Step 8: Random Forest Classifier
│   └── s8_RFC.py                   # Training & evaluating Random Forest model
│
├── RFCmodel/                       # Final Model: Random Forest Classifier
│   └── RFCmodel.py                 # Trains & saves final RFC model as model.pkl
│
├── model/                          # Step 9: Flask web deployment
│   ├── app.py                      # Flask app entry point
│   ├── templates/                  # HTML templates (UI)
│   │   └── index.html
│   └── rfc_model.pkl               # Final saved ML model
│
├── requirements.txt                # Dependencies (pandas, scikit-learn, matplotlib, flask if deploying)
├── sna_aug25.csv                   # Dataset (Age, EstimatedSalary, Purchased)
└── README.md                       # Project documentation


```
---
## 🚀 How to Run the Project
Run the Flask App - Go to the model folder and start the app:

cd ../model python app.py

---

## 📂 You can view the presentation here:  
[Social Network Ads – Presentation](./ML_task2.pptx)

---
## 🙌 Acknowledgement
This project helped me explore different Machine Learning classification models step by step, analyze their performance, and finally deploy the best-performing model (Random Forest Classifier) using Flask.
