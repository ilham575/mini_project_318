# app.py

import csv
import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Load model and scaler
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Update the CSV file path if needed
csv_file_path = 'student_prediction.csv'  # Ensure this path is accessible

def save_to_csv(data):
    file_exists = os.path.isfile(csv_file_path)

    # Write the header only if the file doesn't exist
    try:
        # Get the next Student_ID
        if file_exists:
            existing_data = pd.read_csv(csv_file_path)
            next_student_id = existing_data['Student_ID'].max() + 1
        else:
            next_student_id = 1  # Start from 1 if the file doesn't exist

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Student_ID', 'Attendance', 'Assignment_Completion', 'Midterm_Score', 'Final_Score', 'Result'])
            # Add the next Student_ID to the data
            data_with_id = [next_student_id] + data
            writer.writerow(data_with_id)
            print("บันทึกข้อมูลเรียบร้อยแล้ว.")
    except Exception as e:
        print(f"ข้อผิดพลาดในการบันทึกเป็น CSV: {e}")



@app.route('/', methods=['GET', 'POST'])
def home():
    recommendation = ""  # Initialize recommendation

    if request.method == 'POST':
        # Get data from the form
        attendance = float(request.form['attendance'])
        assignment_completion = float(request.form['assignment_completion'])
        midterm_score = float(request.form['midterm_score'])
        final_score = float(request.form['final_score'])

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Attendance': [attendance],
            'Assignment_Completion': [assignment_completion],
            'Midterm_Score': [midterm_score],
            'Final_Score': [final_score]
        })

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Use the model to predict
        prediction = model.predict(input_data_scaled)

        # Convert prediction to a message
        result = 'Passed' if prediction[0] == 1 else 'Failed'
        
        # Recommendation for those who failed
        if result == 'Failed':
            recommendation = (
                "คุณอาจพิจารณาทบทวนเนื้อหาที่เรียนหรือขอคำแนะนำจากครูผู้สอน "
                "เพื่อพัฒนาทักษะและความเข้าใจในวิชานั้น ๆ "
                "นอกจากนี้ยังสามารถเข้าร่วมกลุ่มศึกษาเพื่อช่วยกันเรียนรู้กับเพื่อนๆ ได้อีกด้วย."
            )

        save_to_csv([attendance, assignment_completion, midterm_score, final_score, result])

        return render_template('result.html', result=result, recommendation=recommendation)

    return render_template('index.html')

# ในฟังก์ชัน predict ในไฟล์ app.py
@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    attendance = request.form['attendance']
    assignment_completion = request.form['assignment_completion']
    midterm_score = request.form['midterm_score']
    final_score = request.form['final_score']
    
    # สร้าง DataFrame เพื่อใช้สำหรับการทำนาย
    input_data = pd.DataFrame({
        'Attendance': [attendance],
        'Assignment_Completion': [assignment_completion],
        'Midterm_Score': [midterm_score],
        'Final_Score': [final_score]
    })

    # ปรับสเกลข้อมูล
    input_data_scaled = scaler.transform(input_data)

    # ใช้โมเดลเพื่อทำการคาดการณ์
    prediction = model.predict(input_data_scaled)

    # แปลงการคาดการณ์เป็นข้อความ
    result = 'Passed' if prediction[0] == 1 else 'Failed'

            # Recommendation for those who failed
    if result == 'Failed':
            recommendation = (
                "คุณอาจพิจารณาทบทวนเนื้อหาที่เรียนหรือขอคำแนะนำจากครูผู้สอน "
                "เพื่อพัฒนาทักษะและความเข้าใจในวิชานั้น ๆ "
                "นอกจากนี้ยังสามารถเข้าร่วมกลุ่มศึกษาเพื่อช่วยกันเรียนรู้กับเพื่อนๆ ได้อีกด้วย."
            )

    

if __name__ == '__main__':
    app.run(debug=True)
