📊 Customer Churn Predictor

Predict telecom customer churn with a Random Forest ML model deployed on Streamlit.
This app allows you to run churn predictions for single customers or entire batches via CSV upload 🚀.

✨ Features

🔍 Single Customer Prediction – Enter customer details manually to get an instant churn prediction.

📂 Batch Upload – Upload a CSV file with customer data and get churn predictions for all rows.

🧠 Automatic Column Mapping – No need to rename columns; the app maps them intelligently.

📊 Interactive Visuals – Probability distribution charts and high-risk customer highlights.

📥 Download Predictions – Export results to CSV for further analysis.

📂 Dataset

This app is powered by the Telco Customer Churn Dataset.

📥 Download Dataset:
Telco Customer Churn – Kaggle

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/your-username/customer-churn-predictor.git
cd customer-churn-predictor

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py

📌 Usage
🔍 Single Customer Mode

Select Single Customer from the sidebar.

Enter values for contract type, payment method, internet service, tenure, monthly charges, and total charges.

Click Predict Churn to see the prediction.

📂 Batch Upload Mode

Select Batch Upload from the sidebar.

Upload a CSV file.

Columns will be automatically mapped, even if names differ slightly.

View predictions, top 10 high-risk customers (with customerID), and download full results.

💡 Tip: For testing, download the dataset from Kaggle:
Telco Customer Churn Dataset

📊 Example Outputs

✅ Stay → The customer is likely to remain.

⚠️ Churn → The customer is at high risk of leaving.

⚡ Tech Stack

Python 3.11+

scikit-learn (Random Forest pipeline)

pandas (data wrangling)

plotly (interactive charts)

Streamlit (web deployment)

🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License.

Would you like me to also add a "Live Demo" section at the top (so when you deploy to Streamlit Cloud, you just drop the link in)?
