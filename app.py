from flask import Flask, render_template, request

app = Flask(__name__)

# Dummy data for preview
people_count = 24
sessions = [
    {'name': 'CS101 Monday', 'date': '2026-05-09'},
    {'name': 'CS101 Wednesday', 'date': '2026-05-07'},
]
recognition_path = {'backend': 'facenet'}

@app.route('/')
def index():
    return render_template('index.html', 
                         people_count=people_count, 
                         sessions=sessions,
                         recognition_path=recognition_path,
                         request=request)

@app.route('/enroll')
def enroll_page():
    return render_template('base.html', 
                         recognition_path=recognition_path,
                         request=request)

@app.route('/live')
def live_page():
    return render_template('base.html',
                         recognition_path=recognition_path,
                         request=request)

@app.route('/dashboard')
def dashboard():
    return render_template('base.html',
                         recognition_path=recognition_path,
                         request=request)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
