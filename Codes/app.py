from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import binascii

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

# Function to authenticate a user
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

# Profile route
@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return "User Profile - Work in progress!"

# Shopping routes
@app.route('/shopping')
def shopping():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('shopping.html', theme=get_theme())

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

@app.route('/amazon')
def amazon():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('amazon.html')  # Ensure you have amazon.html in your templates folder

# Entertainment and games routes
@app.route('/entertainment')
def entertainment():
    if 'username' not in session:
        return redirect(url_for('login_page'), theme=get_theme())
    return "Entertainment page - Work in progress!"

@app.route('/games')
def games():
    if 'username' not in session:
        return redirect(url_for('login_page'), theme=get_theme())
    return "Games page - Work in progress!"

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session:
        return redirect(url_for('login_page'), theme=get_theme())

    username = session['username']
    user_details = get_user_details(username)  # Fetch user details from the file

    if request.method == 'POST':
        # Update user details
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        gender = request.form['gender']
        theme = request.form.get('theme', 'light')  # Default to light theme

        update_user_details(username, name, phone, email, gender, theme)
        flash("Settings updated successfully!")
        return redirect(url_for('settings'))

    # Render settings page with user details
    return render_template('settings.html', user_details=user_details, theme=get_theme())


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


# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login_page'), theme=get_theme())

# Function to get theme preference
def get_theme():
    theme = request.cookies.get('theme', 'light')  # Default to light theme
    return theme



if __name__ == '__main__':
    app.run(debug=True)
