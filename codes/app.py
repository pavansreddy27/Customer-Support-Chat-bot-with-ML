from flask import Flask, jsonify, make_response, render_template, request, redirect, url_for, session, flash
import os
import binascii
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


app = Flask(__name__)

# Generate a random secret key for the application
def generate_secret_key():
    return binascii.hexlify(os.urandom(24)).decode()

app.secret_key = generate_secret_key()  # Secret key for flashing messages

# Path to the TSV file where user details will be stored
TSV_FILE = 'users.tsv'

# Ensure the TSV file exists
if not os.path.exists(TSV_FILE):
    with open(TSV_FILE, 'w') as file:
        pass

# Function to append user data to the TSV file
def write_to_tsv(name, username, password, phone):
    with open(TSV_FILE, 'a') as file:
        file.write(f"{name}\t{username}\t{password}\t{phone}\n")

# Function to check if the username already exists
def check_user_exists(username):
    if os.path.exists(TSV_FILE):
        with open(TSV_FILE, 'r') as file:
            for line in file:
                existing_username = line.split('\t')[1]
                if existing_username.strip() == username:
                    return True
    return False

def authenticate_user(username, password):
    if os.path.exists(TSV_FILE):
        with open(TSV_FILE, 'r') as file:
            for line in file:
                data = line.strip().split('\t')
                if len(data) == 4 and data[1] == username and data[2] == password:
                    return True
    return False

# Home route - Redirects to login page
@app.route('/')
def home():
    return redirect(url_for('login_page'))

# Login page route
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if authenticate_user(username, password):
            session['username'] = username
            return redirect(url_for('dashboard'))  # Redirect to the dashboard upon successful login
        flash("Invalid username or password!")  # Flash error message for invalid login
        return redirect(url_for('login_page'))
    return render_template('login.html')

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        phone = request.form['phone']

        # Check if the username already exists
        if check_user_exists(username):
            flash("Username already exists. Please choose a different one.")
            return redirect(url_for('register'))

        # Write the new user data to the TSV file
        write_to_tsv(name, username, password, phone)
        flash("Registration successful! Please login.")
        return redirect(url_for('login_page'))

    return render_template('register.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    username = session['username']
    return render_template('dashboard.html', username=username, theme=get_theme())


TSV_FILE = "users.tsv"

def fetch_user_details(username):
    try:
        with open(TSV_FILE, 'r') as file:
            lines = file.readlines()
            for line in lines:  # Adjust if there's a header
                user_data = line.strip().split("\t")
                if len(user_data) == 4 and user_data[1] == username:  # Match on username
                    return {"name": user_data[0], "username": user_data[1], "phone": user_data[3]}
        return {"error": "User not found"}
    except FileNotFoundError:
        return {"error": "TSV file not found"}
    except Exception as e:
        return {"error": str(e)}
    
    
@app.route('/profile', methods=['GET'])
def profile():
    username = session.get('username')
    print(f"DEBUG: session username = {username}")  # Debugging line
    if not username:
        return jsonify({"error": "You are not logged in."}), 401

    user_details = fetch_user_details(username)
    return jsonify(user_details)



# Shopping routes
@app.route('/shopping')
def shopping():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('shopping.html', theme=get_theme())

@app.route('/flipkart')
def flipkart():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('flipkart.html')

@app.route('/ajio')
def ajio():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('ajio.html')

@app.route('/myntra')
def myntra():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('myntra.html')

@app.route('/meesho')
def meesho():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('meesho.html')

@app.route('/tatacliq')
def tatacliq():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('tatacliq.html')

  # Ensure you have amazon.html in your templates folder

@app.route('/social_media')
def social_media():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('social_media.html')



@app.route('/entertainment')
def entertainment():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('entertainment.html')



@app.route('/payment')
def payment():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('payment.html')


@app.route('/learning')
def learning():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('learning.html')


@app.route('/food')
def food():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('food.html')


@app.route('/job')
def job():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('job.html')



@app.route('/linkedin')
def linkedin():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('linkedin.html')

@app.route('/apna')
def apna():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('apna.html')

@app.route('/shine')
def shine():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('shine.html')

@app.route('/indeed')
def indeed():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('indeed.html')

@app.route('/naukri')
def naukri():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('naukri.html')


@app.route('/zomato')
def zomato():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('zomato.html')

@app.route('/swiggy')
def swiggy():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('swiggy.html')

@app.route('/ubereats')
def ubereats():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('ubereats.html')

@app.route('/eatsure')
def eatsure():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('eatsure.html')

@app.route('/zepto')
def zepto():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('zepto.html')


@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('chatbot.html')




@app.route('/booking')
def booking():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('booking.html')


@app.route('/instagram')
def instagram():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('instagram.html')

@app.route('/whatsapp')
def whatsapp():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('whatsapp.html')


@app.route('/twitter')
def twitter():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('twitter.html')

@app.route('/snapchat')
def snapchat():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('snapchat.html')


@app.route('/facebook')
def facebook():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('facebook.html')



@app.route('/udemy')
def udemy():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('udemy.html')


@app.route('/coursera')
def coursera():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('coursera.html')


@app.route('/unacademy')
def unacademy():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('unacademy.html')


@app.route('/simplilearn')
def simplilearn():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('simplilearn.html')


@app.route('/nptel')
def nptel():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('nptel.html')

@app.route('/goibibo')
def goibibo():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('goibibo.html')

@app.route('/ixigo')
def ixigo():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('ixigo.html')

@app.route('/bc')
def bc():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('bc.html')

@app.route('/MMT')
def MMT():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('MMT.html')

@app.route('/EMT')
def EMT():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('EMT.html')

@app.route('/amazonprime')
def amazonprime():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('amazonprime.html')

@app.route('/netflix')
def netflix():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('netflix.html')

@app.route('/hotstar')
def hotstar():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('hotstar.html')

@app.route('/jiocinema')
def jiocinema():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('jiocinema.html')

@app.route('/spotify')
def spotify():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('spotify.html')

@app.route('/phonepe')
def phonepe():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('phonepe.html')

@app.route('/gpay')
def gpay():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('gpay.html')

@app.route('/paytm')
def paytm():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('paytm.html')

@app.route('/paypal')
def paypal():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('paypal.html')

@app.route('/razerpay')
def razerpay():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('razerpay.html')


@app.route('/payment_chatbot')
def payment_chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('payment_chatbot.html')

@app.route('/stream_chatbot')
def stream_chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('stream_chatbot.html')

@app.route('/travel_chatbot')
def travel_chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('travel_chatbot.html')

@app.route('/social_media_chatbot')
def social_media_chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('social_media_chatbot.html')

@app.route('/learn_chatbot')
def learn_chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('learn_chatbot.html')

@app.route('/jobs_chatbot')
def jobs_chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('jobs_chatbot.html')

@app.route('/food_chatbot')
def food_chatbot():
    if 'username' not in session:
        return redirect(url_for('login_page'))  # Assuming you have a login page
    return render_template('food_chatbot.html')

# Route for the homepage
@app.route('/amazon')
def amazon():
    return render_template('amazon.html')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session:
        return redirect(url_for('login_page'))

    username = session['username']
    user_details = get_user_details(username)

    if request.method == 'POST':
        # Update user details
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        gender = request.form['gender']
        theme = request.form.get('theme', 'light')

        update_user_details(username, name, phone, email, gender, theme)

        # Set theme cookie
        response = make_response(redirect(url_for('settings')))
        response.set_cookie('theme', theme, max_age=30 * 24 * 60 * 60)  # Cookie expires in 30 days
        print(f"DEBUG: Setting theme cookie with value {theme}")  # Debug line
        flash("Settings updated successfully!")
        return response

    return render_template('settings.html', user_details=user_details)


def get_user_details(username):
    """Fetch user details from the TSV file."""
    if os.path.exists(TSV_FILE):
        with open(TSV_FILE, 'r') as file:
            for line in file:
                data = line.strip().split('\t')
                if len(data) >= 6 and data[1] == username:
                    return {
                        'name': data[0],
                        'username': data[1],
                        'phone': data[3],
                        'email': data[4],
                        'gender': data[5],
                        'theme': data[6] if len(data) > 6 else 'light'  # Default to light theme
                    }
    return None


def update_user_details(username, name, phone, email, gender, theme):
    """Update user details in the TSV file."""
    lines = []
    if os.path.exists(TSV_FILE):
        with open(TSV_FILE, 'r') as file:
            for line in file:
                data = line.strip().split('\t')
                if len(data) >= 6 and data[1] == username:
                    lines.append(f"{name}\t{username}\t{data[2]}\t{phone}\t{email}\t{gender}\t{theme}\n")
                else:
                    lines.append(line)

    with open(TSV_FILE, 'w') as file:
        file.writelines(lines)

@app.route('/logout')
def logout():
    # Remove 'username' from session to log the user out
    session.pop('username', None)
    return redirect(url_for('login_page'))


# Function to get theme preference
def get_theme():
    theme = request.cookies.get('theme', 'light')  # Default to light theme
    return theme

@app.context_processor
def inject_theme():
    return {'theme': get_theme()}

# Chatbot for Shopping Application (Logistic Regression Model)
file_path = "faq_shopping_dataset.csv"
data = pd.read_csv(file_path)
data = data[['Question', 'Answer']]
data = data.sample(n=15, random_state=42)
X = data['Question']
y = data['Answer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
shopping_model = LogisticRegression(max_iter=1000)
shopping_model.fit(X_train_vec, y_train)

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_input = request.form.get('user_input', '').strip()
        if not user_input:
            return "Please provide a valid query.", 400
        print(f"User Input: {user_input}")
        question_vec = vectorizer.transform([user_input])
        response = shopping_model.predict(question_vec)
        print(f"Response: {response[0]}")
        return response[0], 200
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, there was an error processing your request.", 500



# Chatbot for Social Media Application (Naive Bayes Model)
facebook_file_path = "Large_SocialMedia_FAQs.csv"
facebook_data = pd.read_csv(facebook_file_path)
facebook_data = facebook_data[['FAQ', 'Answer']].dropna(subset=['FAQ', 'Answer'])

X_facebook = facebook_data['FAQ']
y_facebook = facebook_data['Answer']
X_facebook_train, X_facebook_test, y_facebook_train, y_facebook_test = train_test_split(X_facebook, y_facebook, test_size=0.2, random_state=42)

facebook_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_facebook_train_vec = facebook_vectorizer.fit_transform(X_facebook_train)
facebook_model = MultinomialNB()
facebook_model.fit(X_facebook_train_vec, y_facebook_train)

def facebook_chatbot_response(question):
    question_vec = facebook_vectorizer.transform([question])
    response = facebook_model.predict(question_vec)
    return response[0] if response else "Sorry, I couldn't find an answer."

@app.route('/facebook_webhook', methods=['POST'])
def facebook_webhook():
    user_message = request.json.get('message')
    if user_message:
        response = facebook_chatbot_response(user_message)
        return jsonify({"message": response})
    return jsonify({"error": "No message found"}), 400


# Chatbot for Learning Application (Semantic Search Model)
learning_data = pd.read_csv("learning.csv")
learning_model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = learning_model.encode(learning_data['Question'].tolist(), convert_to_tensor=True)

@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    try:
        user_input = request.json.get("question")
        if not user_input:
            return jsonify({"error": "No question provided"}), 400
        query_embedding = learning_model.encode(user_input, convert_to_tensor=True)
        scores = cosine_similarity(query_embedding.reshape(1, -1), question_embeddings.numpy())
        best_match_idx = scores.argmax()
        answer = learning_data.iloc[best_match_idx]['Answer']
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Chatbot for Travel Application (Logistic Regression Model)
travel_data = pd.read_csv("travel.csv")
label_map = {answer: idx for idx, answer in enumerate(travel_data['Answer'].unique())}
travel_data['Label'] = travel_data['Answer'].map(label_map)
X_travel = travel_data['Question']
y_travel = travel_data['Label']
X_train_travel, X_test_travel, y_train_travel, y_test_travel = train_test_split(X_travel, y_travel, test_size=0.2, random_state=42)

travel_vectorizer = TfidfVectorizer(max_features=5000)
X_train_travel_tfidf = travel_vectorizer.fit_transform(X_train_travel)
travel_model = LogisticRegression(max_iter=1000)
travel_model.fit(X_train_travel_tfidf, y_train_travel)

@app.route("/handle_travel_query", methods=["POST"])
def handle_travel_query():
    user_input = request.form.get("user_input")
    if user_input:
        question_tfidf = travel_vectorizer.transform([user_input])
        predicted_label = travel_model.predict(question_tfidf)[0]
        predicted_answer = [answer for answer, idx in label_map.items() if idx == predicted_label][0]
        return jsonify({'answer': predicted_answer})
    return "I'm sorry, I didn't understand your question. Please try again."


# Chatbot for Streaming Application (Logistic Regression Model)
entertainment_data = pd.read_csv('entertainment.csv').dropna(subset=['Question', 'Answer'])
tfidf_entertainment = TfidfVectorizer(stop_words='english')
X_entertainment = tfidf_entertainment.fit_transform(entertainment_data['Question'])
y_entertainment = entertainment_data['Answer']
entertainment_model = LogisticRegression()
entertainment_model.fit(X_entertainment, y_entertainment)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.form['user_input']
    user_input_vec = tfidf_entertainment.transform([user_input])
    predicted_answer = entertainment_model.predict(user_input_vec)[0]
    return jsonify({'answer': predicted_answer})


# Chatbot for Payments Application (Logistic Regression Model)
payment_model = joblib.load('logistic_regression_model.pkl')
payment_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/get_chat_response', methods=['POST'])
def get_chat_response():
    user_input = request.form['user_input']
    if not user_input:
        return jsonify({'response': "Please ask a valid question."})
    input_vector = payment_vectorizer.transform([user_input])
    predicted_label = payment_model.predict(input_vector)[0]
    payment_data = pd.read_csv('payment.csv')
    answer = payment_data[payment_data['Answer'] == predicted_label]['Answer'].iloc[0]
    return jsonify({'response': answer})


# Chatbot for Food Ordering Application (Cosine Similarity Model)
food_data = pd.read_csv('food.csv')
food_vectorizer = TfidfVectorizer()
food_vectors = food_vectorizer.fit_transform(food_data['Question'])

@app.route('/get_food_chat_response', methods=['POST'])
def get_food_chat_response():
    user_input = request.form['user_input']
    user_input_vector = food_vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, food_vectors)
    most_similar_index = similarities.argmax()
    response = food_data['Answer'][most_similar_index]
    return jsonify(response)


# Chatbot for Jobs Application (Logistic Regression Model)
job_data = pd.read_csv('job.csv').dropna(subset=['Question', 'Answer'])
X_job = job_data['Question']
y_job = job_data['Answer']
X_train_job, X_test_job, y_train_job, y_test_job = train_test_split(X_job, y_job, test_size=0.2, random_state=42)

job_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
job_model.fit(X_train_job, y_train_job)
joblib.dump(job_model, 'job_question_answer_model.pkl')

@app.route('/get_job_answer', methods=['POST'])
def get_job_answer():
    data = request.get_json()
    user_question = data['question']
    answer = job_model.predict([user_question])[0]
    return jsonify({'answer': answer})

    
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

