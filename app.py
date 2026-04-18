import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, make_response
import csv
import io
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import pickle
import os
from datetime import datetime
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv

# Load environment variables from .env file directly into os.environ
load_dotenv()

app = Flask(__name__)
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)
# Secret key for session management and flash messages
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or '5f352379324c2246ce5982'
# Database Setup (Supports Production Postgres URL or local SQLite)
db_url = os.environ.get('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'sqlite:///diabetes_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail configuration
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')
mail = Mail(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info'

# Load the trained models
try:
    rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
    lr_model = pickle.load(open('models/lr_model.pkl', 'rb'))
    dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
    rf_health_model = pickle.load(open('models/rf_health_model.pkl', 'rb'))
    
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
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    # Relationship to prediction history
    predictions = db.relationship('PredictionHistory', backref='author', lazy=True)

    def get_reset_token(self):
        s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        return s.dumps({'user_id': self.id})

    @staticmethod
    def verify_reset_token(token, expires_sec=1800):
        s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token, max_age=expires_sec)['user_id']
        except:
            return None
        return User.query.get(user_id)

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
        email = request.form.get('email')
        password = request.form.get('password')
        if not username or not email or not password:
            flash('Username, email, and password are required!', 'danger')
            return redirect(url_for('register'))
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
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

def send_reset_email(user):
    token = user.get_reset_token()
    reset_url = url_for('reset_token', token=token, _external=True)
    
    if not app.config['MAIL_USERNAME']:
        print("\n\n" + "="*50)
        print("SIMULATED EMAIL SENT (No MAIL_USERNAME configured)")
        print(f"To: {user.email}")
        print(f"Subject: Password Reset Request")
        print(f"Body: \nTo reset your password, visit the following link:\n{reset_url}")
        print("="*50 + "\n\n")
        return

    msg = Message('Password Reset Request', sender='noreply@healthanalytics.com', recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{reset_url}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    try:
        mail.send(msg)
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            send_reset_email(user)
        flash('If an account exists with that email, an email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html')

@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html')

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

    explanation = ""
    try:
        model_core = selected_model.named_steps['classifier'] if hasattr(selected_model, 'named_steps') else selected_model
        if hasattr(model_core, 'feature_importances_'):
            importances = model_core.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            top_3 = [f[0] for f in feature_imp[:3]]
            explanation = f"Top Contributing Factors: {', '.join(top_3)}."
    except Exception as e:
        pass

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

    return render_template('index.html', prediction_text=f'Patient has a {result_text}.', explanation=explanation, selected_model=selected_model_name, dataset_choice=dataset_choice)

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

@app.route('/export_history')
@login_required
def export_history():
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.date_posted.desc()).all()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Model Used', 'Input Data', 'Result'])
    
    for p in predictions:
        cw.writerow([p.date_posted.strftime('%Y-%m-%d %H:%M:%S'), p.model_used, p.input_data, p.prediction_result])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=medical_history_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output

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
    app.run(debug=True, host='0.0.0.0')
