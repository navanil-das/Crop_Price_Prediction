# 🌾 Crop_Price_Prediction

A Machine Learning web application that predicts the market price of agricultural crops based on multiple factors such as crop type, state, district, market conditions, and historical data.

The project combines data preprocessing, machine learning modeling, and an interactive Streamlit UI to allow users to easily estimate crop prices.

🚀 Live Demo

If deployed on Streamlit Cloud:

https://your-app-name.streamlit.app
📌 Features

📊 Crop Price Prediction using Machine Learning

🌍 Location-based prediction (State, District, Market)

🌱 Crop selection input

⚙️ Advanced settings panel

📈 Interactive Streamlit dashboard

🧠 Trained ML model for price prediction

🖥️ User-friendly interface

🧠 Machine Learning Workflow

Data Collection

Data Cleaning

Feature Engineering

Encoding Categorical Variables

Model Training

Model Evaluation

Model Deployment with Streamlit

🛠️ Tech Stack
Programming

Python

Machine Learning

Scikit-learn

Pandas

NumPy

Visualization / UI

Streamlit

Deployment

Streamlit Cloud

GitHub

📂 Project Structure
crop_price_prediction/
│
├── app/
│   └── streamlit_app.py        # Streamlit user interface
│
├── models/
│   └── trained_model.pkl       # Saved ML model
│
├── data/
│   └── crop_data.csv           # Dataset used for training
│
├── notebooks/
│   └── model_training.ipynb    # Model experimentation
│
├── requirements.txt            # Python dependencies
│
└── README.md                   # Project documentation
⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/crop_price_prediction.git
cd crop_price_prediction

Create virtual environment (recommended):

python -m venv venv

Activate environment

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate

Install dependencies

pip install -r requirements.txt
▶️ Running the Application

Run the Streamlit app locally:

streamlit run app/streamlit_app.py

Then open in browser:

http://localhost:8501
☁️ Deployment (Streamlit Cloud)

Push the project to GitHub

Go to Streamlit Cloud

Connect your GitHub repository

Select:

app/streamlit_app.py

Add requirements.txt

Streamlit will automatically install dependencies and deploy the app.

📊 Example Inputs
Input	Example
Crop	Rice
State	West Bengal
District	Hooghly
Market	Arambagh

Output:

Predicted Price: ₹XXXX per quintal
📈 Future Improvements

Add more ML models (XGBoost, LightGBM)

Improve prediction accuracy

Add real-time market price APIs

Build historical price visualization

Add mobile-friendly UI

🤝 Contributing

Contributions are welcome.

Steps:

Fork the repository

Create a new branch

Commit changes

Submit a pull request

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Navanil Das

Machine Learning & Cybersecurity Enthusiast
