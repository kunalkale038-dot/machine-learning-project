🛒 Super Store Sales Analytics & Prediction
A Machine Learning-powered Super Store Sales Analytics & Prediction Dashboard built using Python, Scikit-Learn, KNN Regression, and Streamlit.

The application predicts Item Outlet Sales based on product and outlet-related features and provides an interactive dashboard for single predictions, batch CSV predictions, and model insights.

🚀 Features
🔮 Single Sales Prediction

Enter product/outlet feature values
Generate estimated Item Outlet Sales
Display prediction in an interactive dashboard
📊 Batch CSV Prediction

Upload a CSV file
Generate predictions for multiple records
View prediction distribution
Download prediction results as CSV
🤖 KNN Regression Model

K-Nearest Neighbors Regression
K = 7
Manhattan distance (p = 1)
MinMaxScaler for feature normalization
📈 Model Insights

View model configuration
Display number of required features
Visualize feature sensitivity
🎨 Modern Dashboard

Streamlit interface
Responsive layout
Interactive Plotly visualizations
Prediction cards and metrics
🧠 Machine Learning Workflow
The project follows a standard Machine Learning workflow:

Dataset
   ↓
Data Preprocessing
   ↓
Feature Encoding
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
KNN Regression
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Dashboard
   ↓
Sales Prediction
🛠️ Technologies Used
Technology	Purpose
Python	Programming
Pandas	Data processing
NumPy	Numerical operations
Scikit-Learn	Machine Learning
KNN Regression	Sales prediction
MinMaxScaler	Feature scaling
Streamlit	Web dashboard
Plotly	Interactive visualization
Pickle	Saving trained model
📂 Project Structure
Super-Store-Sales-Prediction/
│
├── app.py
├── KNN Regression(1).ipynb
├── KNN_reg_outlet_sales.csv
│
├── knn_regression_model.pkl
├── minmax_scaler.pkl
├── features.pkl
│
├── requirements.txt
└── README.md
📊 Dataset
The project uses a Super Store / Outlet Sales dataset containing product and outlet-related attributes.

The prediction target is:

Item Outlet Sales
The model uses processed numerical features generated during the data preprocessing stage.

🤖 Model
K-Nearest Neighbors Regression
The project uses KNN Regression to estimate Item Outlet Sales.

Model configuration:

Algorithm       : KNN Regression
K Neighbors     : 7
Distance Metric : Manhattan
p               : 1
Scaler          : MinMaxScaler
Target          : Item Outlet Sales
The trained model is stored in:

knn_regression_model.pkl
The trained scaler is stored in:

minmax_scaler.pkl
The feature list is stored in:

features.pkl
📊 Dashboard Modules
1. Single Prediction
Users can enter feature values through the Streamlit interface.

The application:

User Input
   ↓
DataFrame
   ↓
MinMaxScaler
   ↓
KNN Model
   ↓
Predicted Sales
The predicted sales value is displayed in the dashboard.

2. Batch CSV Prediction
Users can upload a CSV containing the required feature columns.

The application:

Reads the uploaded CSV.
Checks required features.
Applies the saved scaler.
Generates predictions using the trained KNN model.
Adds a Predicted_Sales column.
Displays summary metrics.
Shows a prediction distribution chart.
Allows the user to download the prediction results.
3. Model Insights
The dashboard provides information about:

KNN algorithm
Number of neighbors
Distance metric
Feature count
Feature sensitivity visualization
⚙️ Installation
Clone the repository:

git clone https://github.com/YOUR_USERNAME/Super-Store-Sales-Prediction.git
Move into the project directory:

cd Super-Store-Sales-Prediction
Create a virtual environment:

py -m venv venv
Activate it on Windows:

venv\Scripts\activate
Install dependencies:

py -m pip install -r requirements.txt
▶️ Run the Application Locally
Run:

py -m streamlit run app.py
The application will normally open at:

http://localhost:8501
📦 Requirements
Create a requirements.txt file containing:

streamlit
pandas
numpy
scikit-learn
plotly
☁️ Free Deployment
This application can be deployed using Streamlit Community Cloud.

Basic deployment steps:

GitHub Repository
       ↓
Upload Project Files
       ↓
Open Streamlit Community Cloud
       ↓
Select Repository
       ↓
Select app.py
       ↓
Deploy
       ↓
Live Web Application
Make sure the repository contains:

app.py
requirements.txt
knn_regression_model.pkl
minmax_scaler.pkl
features.pkl
📸 Dashboard
The application provides an interactive dashboard containing:

Super Store Sales Prediction
Single Prediction
Batch Prediction
Model Insights
Prediction Metrics
Interactive Charts
CSV Download
🔮 Future Improvements
Possible future enhancements include:

📌 Add authentication/login
📌 Improve categorical feature selection
📌 Add historical sales analysis
📌 Add advanced interactive charts
📌 Compare KNN with Random Forest and XGBoost
📌 Add model performance dashboard
📌 Add automated data validation
📌 Add cloud database integration
📌 Improve mobile responsiveness
🎯 Project Objective
The main objective of this project is to build an interactive Machine Learning application that can estimate Item Outlet Sales and provide useful analytics through a user-friendly dashboard.

This project demonstrates the complete workflow from:

Data → Preprocessing → Machine Learning → Model Saving → Prediction → Dashboard → Deployment

👨‍💻 Author
kunal kale

B.Tech — Artificial Intelligence & Data Science

⭐ If You Like This Project
If you find this project useful, consider giving the repository a ⭐ on GitHub.

📜 License
This project is created for educational and learning purposes.
