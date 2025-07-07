from flask import Flask, render_template, Response,jsonify, request
import cv2
import importlib
import share_state

app = Flask(__name__)

cap = cv2.VideoCapture(0)
selected_exercise = "bicep_curl"


def process_video():
    global selected_exercise
    exercise_module = importlib.import_module(selected_exercise)  # Dynamically load module
    return exercise_module.process_video(cap)  # Run function from module

@app.route('/')
def index():
    return render_template('index.html')  # Serve the web dashboard

@app.route('/video_feed')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_exercise', methods=['POST'])
def select_exercise():
    global selected_exercise
    selected_exercise = request.form['exercise']
    return '', 204  # No response needed

@app.route('/get_reps')
def get_reps():
    global selected_exercise
    exercise_module = importlib.import_module(selected_exercise)
    return jsonify(exercise_module.get_reps())

tracking_enabled = False  # Global toggle

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    share_state.tracking_enabled = True
    return '', 204

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    share_state.tracking_enabled = False
    return '', 204

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
