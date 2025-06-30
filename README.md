# Diabetes Prediction App

A production-ready web application for diabetes prediction using machine learning, deployed on Streamlit Cloud.

## Features
- Interactive UI with real-time predictions
- Model accuracy: ~86.6%
- Feature importance visualization
- Confusion matrix and performance metrics
- Cross-platform support (Windows/macOS/Linux)

## Prerequisites
- Python 3.10+
- pip (Python package manager)
- Git (for cloning the repository)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/satvik-05/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Set Up Virtual Environment

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will be available at: http://localhost:8501

## Deployment

### Streamlit Cloud
1. **Fork** this repository to your GitHub account
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and select your forked repository
4. Set the branch to `main`
5. Set the main file path to `app.py`
6. Click "Deploy!"

## Project Structure
```
.
├── .env.example           # Example environment variables
├── .gitignore            # Git ignore file
├── .streamlit/            # Streamlit configuration
│   └── config.toml       # Streamlit settings
├── LICENSE               # License file
├── README.md             # This file
├── app.py               # Main application file
├── diabetes.csv          # Sample dataset
├── models/               # Trained models
│   └── model.joblib     # Pre-trained model
├── requirements.txt      # Project dependencies
└── setup.py             # Package configuration
```

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information. full details.

Copyright (c) 2025 Satvik Jangala
