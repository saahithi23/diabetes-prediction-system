import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import pickle
import os
from datetime import datetime

app = Flask(__name__)
# Secret key for session management and flash messages
app.config['SECRET_KEY'] = '5f352379324c2246ce5982'
# SQLite database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diabetes_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info'

# Load the trained models
try:
    rf_model = pickle.load(open('rf_model.pkl', 'rb'))
    lr_model = pickle.load(open('lr_model.pkl', 'rb'))
    dt_model = pickle.load(open('dt_model.pkl', 'rb'))
    rf_health_model = pickle.load(open('rf_health_model.pkl', 'rb'))
    
    models = {
        'Random Forest (PIMA)': rf_model,
        'Logistic Regression (PIMA)': lr_model,
        'Decision Tree (PIMA)': dt_model,
        'Random Forest (Lifestyle)': rf_health_model
    }
except Exception as e:
    print("Error loading models. Did you run 'python model.py' and 'python train_health_model.py' yet?")
    raise e

# --- Database Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    # Relationship to prediction history
    predictions = db.relationship('PredictionHistory', backref='author', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    model_used = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.String(200), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    # Foreign key linking to user id
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return redirect(url_for('register'))
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        try:
            db.session.commit()
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@app.route('/home')
@login_required
def home():
    return render_template('index.html', selected_model="Random Forest")

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    dataset_choice = request.form.get('dataset_choice', 'pima')
    
    if dataset_choice == 'pima':
        selected_model_name = request.form.get('model_choice_pima', 'Random Forest (PIMA)')
        selected_model = models.get(selected_model_name, rf_model)
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    else:
        selected_model_name = 'Random Forest (Lifestyle)'
        selected_model = rf_health_model
        feature_names = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'Age']
    
    try:
        int_features = [float(request.form.get(f, 0)) for f in feature_names]
    except Exception as e:
        flash('Invalid input! Please enter only valid numeric values for all fields.', 'danger')
        return redirect(url_for('home'))
        
    final_features = [np.array(int_features)]
    prediction = selected_model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1 :
        result_text = "high risk of diabetes"
    else:
        result_text = "low risk of diabetes"

    # Save to database history
    input_str = ", ".join([str(x) for x in int_features])
    history_record = PredictionHistory(
        model_used=selected_model_name, 
        input_data=input_str, 
        prediction_result=result_text, 
        author=current_user
    )
    db.session.add(history_record)
    db.session.commit()

    return render_template('index.html', prediction_text=f'Patient has a {result_text}.', selected_model=selected_model_name, dataset_choice=dataset_choice)

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.date_posted.desc()).all()
    
    diabetic_count = sum(1 for p in predictions if 'diabetic patient' in p.prediction_result or 'high risk' in p.prediction_result)
    non_diabetic_count = len(predictions) - diabetic_count
    
    rf_count = sum(1 for p in predictions if 'Random Forest' in p.model_used and 'PIMA' in p.model_used or p.model_used == 'Random Forest')
    lr_count = sum(1 for p in predictions if 'Logistic Regression' in p.model_used)
    dt_count = sum(1 for p in predictions if 'Decision Tree' in p.model_used)
    lf_count = sum(1 for p in predictions if 'Lifestyle' in p.model_used)
    
    stats = {
        'diabetic': diabetic_count,
        'non_diabetic': non_diabetic_count,
        'rf': rf_count,
        'lr': lr_count,
        'dt': dt_count,
        'lf': lf_count,
        'total': len(predictions)
    }
    return render_template('dashboard.html', predictions=predictions, stats=stats)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls without auth
    '''
    data = request.get_json(force=True)
    prediction = rf_model.predict([np.array(list(data.values()))])
    output = int(prediction[0])
    return jsonify(output)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)
