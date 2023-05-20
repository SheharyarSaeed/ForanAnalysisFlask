from flask import Flask, request, jsonify, send_from_directory, g, session, redirect, url_for, abort
import sqlite3
import os
from ForanAnalysisModel import predict_sentiment, GetCleanedText, getScrapperData
from ForanAnalysisModel2 import predict_sentiment2
import pandas as pd
import numpy as np
from contextlib import closing
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import json

app = Flask(__name__, static_folder='ForanAnalysis/build', static_url_path='')

DATABASE = 'FADataBase.db'
UPLOAD_FOLDER = 'usersImages/uploads'
app.secret_key = '4C173987CECC6233AE2E505D'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the database tables if they don't exist
# def init_db():
#     with closing(connect_db()) as db:
#         with app.open_resource('schema.sql', mode='r') as f:
#             db.cursor().executescript(f.read())
#         db.commit()


# Connect to the database
def connect_db():
    return sqlite3.connect(DATABASE)

# Connect to the SQLite database
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


# Create the users table if it doesn't exist
def init_db():
     with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT NOT NULL,
                       password TEXT NOT NULL,
                       email TEXT NOT NULL UNIQUE,
                       profile_img TEXT)''')
    
        cursor.execute('''CREATE TABLE IF NOT EXISTS history
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id INTEGER,
                       timestamp TEXT NOT NULL,
                       input_data TEXT NOT NULL,
                       type TEXT NOT NULL,
                       FOREIGN KEY (user_id) REFERENCES users(id))''')
        db.commit() 

# Create a user and insert into the users table
def create_user(username,email, password):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("INSERT INTO users (username, email, password) VALUES (?,?, ?)",
                   (username,email, password))

    db.commit()

# Authenticate a user
def authenticate_user(email, password):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?",
                   (email, password))

    user = cursor.fetchone()
    if user:
        return dict(user)
    else:
        return None

# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if username and email and password:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            #abort(401, 'email already exists') 
            return jsonify({'message': 'email already exists'}), 400

        create_user(username,email, password)
        user = authenticate_user(email, password)
        session['user_id'] = user['id']
        return jsonify({'message': 'User created successfully'}), 200
    else:
        # abort(401, 'Invalid data') 
        return jsonify({'message': 'Invalid data'}), 400

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if email and password:
        user = authenticate_user(email, password)

        if user:
            session['user_id'] = user['id']
            return jsonify({'message': 'Logged in successfully'}), 200
        else:
            return jsonify({'message': 'Invalid credentials'}) , 401
    else:
        return jsonify({'message': 'Invalid data'}), 400

@app.route('/api/check-login', methods=['GET'])
def check_login():
    if 'user_id' not in session:
        return jsonify({'loggedIn': False}), 401
    else:
        return jsonify({'loggedIn': True}),200

# Logout route
@app.route('/logout')
def logout():
    if 'user_id' not in session:
        return jsonify({'message': 'Already Logged Out'})
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'})

# Upload profile image route
@app.route('/upload-profile-img', methods=['POST'])
def upload_profile_img():
    if 'user_id' in session:
        user_id = session['user_id']
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            if filename != '':
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # Update the user's profile_img field in the database
                db = get_db()
                cursor = db.cursor()
                cursor.execute("UPDATE users SET profile_img = ? WHERE id = ?",
                               (filename, user_id))
                db.commit()

                return
            
# Get user profile information functionality
@app.route('/profile', methods=['GET'])
def get_profile():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'User is not logged in!'}), 401

    user_id = session['user_id']

    # Retrieve user's profile information from the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, username, email, profile_img, password FROM users WHERE id=?", (user_id,))
    user_data = cursor.fetchone()

    # Check if the user exists
    if user_data is None:
        return jsonify({'error': 'User not found!'}), 404

    # Format the user profile data
    profile = {
        'id': user_data[0],
        'username': user_data[1],
        'email': user_data[2],
        'profile_img': user_data[3],
        'password': user_data[4]
    }

    return jsonify({'profile': profile}), 200


# Change profile data functionality
@app.route('/profile', methods=['PUT'])
def change_profile():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'User is not logged in!'}), 401

    # Get data from request
    db = get_db()
    cursor = db.cursor()
    email = request.json.get('email', None)
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    profile_img = request.json.get('profile_img', None)

    user_id = session['user_id']

    # Update the user's profile data in the database
    cursor.execute("UPDATE users SET email=?, username=?, password=?, profile_img=? WHERE id=?",
              (email, username, password, profile_img, user_id))
    db.commit()

    # Update the user's data in the session
    session['username'] = username
    session['profile_img'] = profile_img

    profile = {
        'id': user_id,
        'username': username,
        'email': email,
        'profile_img': profile_img,
        'password': password
    }

    return jsonify({'message': 'Profile data updated successfully!','profile': profile}), 200

 

# Get all history items for a user functionality
@app.route('/history/all', methods=['GET'])
def get_all_history():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'User is not logged in!'}), 401

    user_id = session['user_id']

    # Retrieve all history items for the user from the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM history WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
    history = cursor.fetchall()

    # Format the history data
    history_data = []
    for entry in history:
        history_data.append({
            'id': entry[0],
            'user_id': entry[1],
            'timestamp': entry[2],
            'input_data': entry[3],
            'type': entry[4]
        })

    return jsonify({'history': history_data}), 200


# Add item to user's history functionality
@app.route('/history/add', methods=['POST'])
def add_to_history():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'User is not logged in!'}), 401

    user_id = session['user_id']

    # Get data from request
    timestamp = request.json.get('timestamp', '')
    input_data = request.json.get('input_data', '')
    item_type = request.json.get('type', '')

    # Add the item to the user's history in the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO history (user_id, timestamp, input_data, type)
        VALUES (?, ?, ?, ?)
    """, (user_id, timestamp, input_data, item_type))
    db.commit()

    return jsonify({'message': 'Item added to history successfully!'}), 200


# Perform database migration
def perform_migration():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        # cursor.execute("PRAGMA table_info(history)")
        # columns = cursor.fetchall()
        # column_names = [column[1] for column in columns]
        # if 'type' not in column_names:
        #     cursor.execute("ALTER TABLE users ADD COLUMN type TEXT NOT NULL")
        #     db.commit()
        cursor.execute("CREATE TABLE users_temp AS SELECT * FROM users")  # Create a temporary table
        cursor.execute("DROP TABLE users")  # Drop the original table
        cursor.execute("CREATE TABLE users AS SELECT * FROM users_temp")  # Recreate the original table
        cursor.execute("DROP TABLE users_temp")  # Drop the temporary table
        cursor.execute("CREATE UNIQUE INDEX idx_unique_email ON users (email)")  # Add unique constraint on email column


# @app.route("/api/predict_sentiment/WordCloud",  methods=['POST'])
def generate_wordcloud(df):
    # Get text data from the request
    # data = request.json
    # text = data['text']
    
    # Generate word cloud
    text_data = df[df.columns[0]].tolist()
    text = ' '.join(text_data)
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    # Convert word cloud image to base64
    img_buffer = io.BytesIO()
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Return the base64 image as JSON response
    return img_str

# Your sentiment analysis API endpoint
@app.route('/api/predict_sentiment', methods=['POST'])
def predict_sentiment_api():
    text = request.json['text']

    if(request.json['type']=="sentence"):
        # scrapperData = text
        data = {'Post': [text]}
        scrapperData = pd.DataFrame(data)
        DataAry = np.asarray(scrapperData["Post"])
        textarray = GetCleanedText(DataAry)
        result = predict_sentiment(textarray)
        multiResult = predict_sentiment2(textarray)
        # Merge multiple dataframes
        df1 = textarray
        # df1.columns = ["Post"]
        df1['index'] = df1.index
        df1['URL'] = "noURL"
        
    elif(request.json['type']=="reddit"):
        scrapperData = getScrapperData(text)
        scrapperData["Post"] = scrapperData[['Title', 'SelfText']].agg(' '.join, axis=1)
        DataAry = np.asarray(scrapperData["Post"])
        textarray = GetCleanedText(DataAry)
        result = predict_sentiment(textarray)
        multiResult = predict_sentiment2(textarray)
        # Merge multiple dataframes
        df1 = textarray
        # df1.columns = ["Post"]
        df1['index'] = df1.index
        df1['URL'] = scrapperData["URL"]
    
    df2 = pd.DataFrame( 
        result ,
        columns=["Negative", "Neutral", "Positive"])
    df2['index'] = df2.index

    df3 = pd.DataFrame( 
        multiResult,
        columns=['toxic',  'severe_toxic',  'obscene'  ,'threat',  'insult' , 'identity_hate'])
    df3['index'] = df3.index

    df2 = df2[['Positive', 'Neutral', 'Negative', 'index']]
    Result =  pd.merge(pd.merge(df1,df2,on='index'),df3,on='index' )
    Result = Result.drop('index', axis=1)
    columns = Result.columns.tolist()
    columns.remove('URL')
    columns.append('URL')
    Result = Result[columns]
    Resultjson = Result.to_json(orient='split')
    result_dict = json.loads(Resultjson)

    # Add item in users history
    if 'user_id' not in session:
        return redirect(url_for('Register'))

    date = datetime.now()

    user_id = session['user_id']
    timestamp = date
    input_data = request.json['text']
    item_type = request.json['type']

    #create wordcloud image
    wordcloudimg = generate_wordcloud(textarray)

    # Add the item to the user's history in the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO history (user_id, timestamp, input_data, type)
        VALUES (?, ?, ?, ?)
    """, (user_id, timestamp, input_data, item_type))
    db.commit()

    return jsonify({'result': result_dict, 'wordcloud':wordcloudimg})

    

# Serve the React build files
@app.route('/', defaults={'path': ''})
@app.route('/Register', defaults={'path': '/Register'} )
@app.route('/Dashboard', defaults={'path': '/Dashboard'})
@app.route('/<path:path>')
def serve_react_app(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        # Serve static files from the 'static' folder
        return send_from_directory(app.static_folder, path)
    else:
        # For all other routes, serve the index.html file
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # perform_migration()
    init_db() # Create the tables if they don't exist
    app.run()


# asset-manifest.json
