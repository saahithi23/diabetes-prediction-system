# Full-Stack Diabetes Risk Prediction System

A professional, full-stack machine learning application built with Flask that predicts a user's risk of developing diabetes. This system provides two separate prediction pipelines: one based on clinical medical data and another based on general lifestyle health indicators. 

The project features a sleek, modern glassmorphic UI, user authentication, and a personal dashboard to track historical predictions.

## 🌟 Key Features

- **Dual-Model Prediction System**: 
  - **Clinical Model**: Uses the PIMA Indians Diabetes Dataset (focusing on metrics like Glucose, BMI, Insulin, and Skin Thickness).
  - **Lifestyle Model**: Uses the CDC's BRFSS2015 Health Indicators Dataset (focusing on daily habits like Physical Activity, Diet, Smoking, and General Health).
- **Realistic Machine Learning Implementation**: Data leakage and overfitting issues found in typical beginner tutorials have been explicitly handled. The models achieve a robust, realistic ~84% test accuracy by strictly preventing pre-split data leakage and tuning the hyperparameters.
- **Algorithm Comparison**: Users can dynamically select between Random Forest, Logistic Regression, and Decision Tree algorithms for the clinical dataset.
- **Modern UI/UX**: Fully responsive, dark-mode glassmorphic interface providing a premium aesthetic.
- **Secure Authentication**: User accounts are protected with `Flask-Login`, `Flask-Bcrypt` password hashing, and **Email-Based Password Reset** using time-sensitive cryptographic tokens (`itsdangerous`).
- **Prediction History Dashboard**: Users can view their past inputs and prediction results, powered by a `SQLAlchemy` relational database.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, SQLAlchemy, Flask-Login, Flask-Bcrypt
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism Design System)

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-link>
   cd prediction-of-diabetes-master
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On Mac/Linux use: source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Train the ML Models**
   Generate fresh, localized `.pkl` model files by running the training scripts from the project root:
   ```bash
   # Train the clinical PIMA models
   python scripts/model.py

   # Train the lifestyle BRFSS model
   python scripts/train_health_model.py
   ```

4. **Set Environment Variables (Optional)**
   For the email capabilities to function properly, set the following environment variables (locally or in your deployment provider):
   - `EMAIL_USER`: Your email address
   - `EMAIL_PASS`: Your app-specific password

5. **Run the Application Locally**
   ```bash
   python app.py
   ```
   The application will start the development server locally at `http://127.0.0.1:5000`.

## ☁️ Cloud Deployment

The project is fully prepared for cloud deployment. A `Procfile` and `gunicorn` framework are included for platforms like Render or Heroku.

1. Push your repository to your GitHub account.
2. Link your GitHub to Render.com (or equivalent) and create a **Web Service**.
3. Set the start command to `gunicorn app:app`.
4. Ensure you define any necessary Environment Variables on your hosting provider.
5. Deploy safely. No database migrations needed; the local SQL instances will dynamically rebuild.

## 🧠 Handling Data Leakage (A Note on Accuracy)
Unlike many beginner examples that artificially boast a 99% accuracy on the PIMA dataset due to aggressive over-sampling (e.g., SMOTE applied *before* standard train/test splits), this project actively addresses the resulting **data leakage**. Duplicate and synthetic rows are stripped prior to splitting, ensuring the Logistic Regression and Random Forest algorithms are evaluated on strictly unseen data, yielding a reliable and honest **~84% test accuracy**.

## 📄 License
This project is built for educational and portfolio demonstration purposes.
