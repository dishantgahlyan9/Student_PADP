import os
import io
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = "student_dropout_secret_key"

# Path to the exported model bundle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'student_model.pkl')

def load_model_bundle():
    """Loads model bundle if available."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def engineer_features(df):
    """
    Applies the exact same feature engineering used during notebook training.
    Works for both a single row and a batch dataframe.
    """
    df = df.copy()

    # Drop Student_ID if present in input
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
    if 'Dropout' in df.columns:
        df = df.drop(columns=['Dropout'])

    # 1. Capture missing values count
    null_cols = [c for c in ['Family_Income', 'Study_Hours_per_Day', 'Stress_Index'] if c in df.columns]
    df['Null_Count'] = df[null_cols].isnull().sum(axis=1)

    # 2. Ordinal mapping for Semester
    if 'Semester' in df.columns:
        df['Semester_Num'] = df['Semester'].astype(str).str.replace('Year ', '', regex=False)
        df['Semester_Num'] = pd.to_numeric(df['Semester_Num'], errors='coerce').fillna(1).astype(int)
    else:
        df['Semester_Num'] = 1

    # Ensure numeric columns are properly typed
    numeric_cols = [
        'Age', 'Family_Income', 'Study_Hours_per_Day', 'Attendance_Rate',
        'Assignment_Delay_Days', 'Travel_Time_Minutes', 'Stress_Index',
        'GPA', 'Semester_GPA', 'CGPA'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Median proxies for interaction ratios
    study_temp = df['Study_Hours_per_Day'].fillna(4.0)
    stress_temp = df['Stress_Index'].fillna(5.5)

    # 4. Interaction & trajectory features
    df['GPA_Trend'] = df['Semester_GPA'] - df['GPA']
    df['CGPA_Diff'] = df['Semester_GPA'] - df['CGPA']
    df['Study_vs_Travel'] = study_temp / (df['Travel_Time_Minutes'] / 60.0 + 0.1)
    df['Attendance_Disruption'] = (100.0 - df['Attendance_Rate']) * df['Assignment_Delay_Days']
    df['Stress_Study_Ratio'] = stress_temp / (study_temp + 0.1)
    df['Academic_Risk_Index'] = (4.0 - df['GPA']) * (100.0 - df['Attendance_Rate'])

    return df

def predict_dataframe(df_raw, bundle):
    """Transforms raw features and predicts dropout probability and label."""
    preprocessor = bundle['preprocessor']
    selector = bundle.get('selector', None)
    model = bundle['model']
    threshold = bundle.get('threshold', 0.50)

    # 1. Feature engineering
    df_engineered = engineer_features(df_raw)

    # 2. Transform through preprocessing pipeline
    transformed = preprocessor.transform(df_engineered)

    # 3. Feature selection (if rfecv was used)
    if selector is not None:
        selected = selector.transform(transformed)
    else:
        selected = transformed

    # 4. Probability & Prediction using optimal threshold
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(selected)[:, 1]
    else:
        probabilities = model.predict(selected).astype(float)

    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


@app.route('/', methods=['GET', 'POST'])
def home():
    bundle = load_model_bundle()
    single_result = None
    batch_results = None
    active_tab = 'single'

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if bundle is None:
            flash("Model file 'student_model.pkl' not found. Please export your model from the notebook first.", "danger")
            return render_template('index.html', single_result=None, batch_results=None, active_tab='single')

        # ----------------- CASE 1: Single Student Input -----------------
        if form_type == 'single':
            active_tab = 'single'
            try:
                # Gather inputs from form
                data = {
                    'Age': [float(request.form.get('Age', 20))],
                    'Gender': [request.form.get('Gender', 'Male')],
                    'Family_Income': [float(request.form.get('Family_Income', 35000))],
                    'Internet_Access': [request.form.get('Internet_Access', 'Yes')],
                    'Study_Hours_per_Day': [float(request.form.get('Study_Hours_per_Day', 4.0))],
                    'Attendance_Rate': [float(request.form.get('Attendance_Rate', 85.0))],
                    'Assignment_Delay_Days': [float(request.form.get('Assignment_Delay_Days', 1))],
                    'Travel_Time_Minutes': [float(request.form.get('Travel_Time_Minutes', 30.0))],
                    'Part_Time_Job': [request.form.get('Part_Time_Job', 'No')],
                    'Scholarship': [request.form.get('Scholarship', 'No')],
                    'Stress_Index': [float(request.form.get('Stress_Index', 5.0))],
                    'GPA': [float(request.form.get('GPA', 2.5))],
                    'Semester_GPA': [float(request.form.get('Semester_GPA', 2.5))],
                    'CGPA': [float(request.form.get('CGPA', 2.5))],
                    'Semester': [request.form.get('Semester', 'Year 1')],
                    'Department': [request.form.get('Department', 'CS')]
                }

                df_single = pd.DataFrame(data)
                pred, proba = predict_dataframe(df_single, bundle)

                risk_score = round(proba[0] * 100, 1)
                is_dropout = bool(pred[0] == 1)

                single_result = {
                    'is_dropout': is_dropout,
                    'risk_score': risk_score,
                    'status': 'High Risk - Potential Dropout' if is_dropout else 'Low Risk - Likely to Continue',
                    'badge_class': 'danger' if is_dropout else 'success',
                    'inputs': request.form
                }
            except Exception as e:
                flash(f"Error processing single prediction: {str(e)}", "danger")

        # ----------------- CASE 2: Batch CSV Upload -----------------
        elif form_type == 'batch':
            active_tab = 'batch'
            file = request.files.get('csv_file')

            if not file or file.filename == '':
                flash("Please select a valid CSV file to upload.", "warning")
            else:
                try:
                    df_upload = pd.read_csv(file)
                    preds, probas = predict_dataframe(df_upload, bundle)

                    df_upload['Dropout_Prediction'] = np.where(preds == 1, 'Likely Dropout', 'Likely Continue')
                    df_upload['Dropout_Probability (%)'] = (probas * 100).round(2)
                    df_upload['Risk_Level'] = np.where(probas >= 0.70, 'High', np.where(probas >= 0.45, 'Medium', 'Low'))

                    # Store for preview in table
                    preview_data = df_upload.head(50).to_dict(orient='records')
                    columns = df_upload.columns.tolist()

                    # Cache CSV for download
                    csv_buffer = io.StringIO()
                    df_upload.to_csv(csv_buffer, index=False)
                    app.config['LAST_BATCH_CSV'] = csv_buffer.getvalue()

                    batch_results = {
                        'total_rows': len(df_upload),
                        'high_risk_count': int((preds == 1).sum()),
                        'columns': columns,
                        'rows': preview_data
                    }
                    flash(f"Successfully processed {len(df_upload)} student records!", "success")
                except Exception as e:
                    flash(f"Error reading or processing CSV: {str(e)}", "danger")

    return render_template('index.html', single_result=single_result, batch_results=batch_results, active_tab=active_tab)


@app.route('/download-predictions')
def download_predictions():
    """Allows downloading the processed batch prediction CSV."""
    csv_data = app.config.get('LAST_BATCH_CSV', None)
    if not csv_data:
        flash("No batch predictions available for download.", "warning")
        return redirect(url_for('home'))

    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='student_predictions.csv'
    )
@app.route('/guide')
def guide():
    """Renders the dedicated feature documentation and dataset guide page."""
    return render_template('guide.html')
@app.route('/download-sample')
def download_sample_csv():
    """Generates and downloads a ready-to-use sample CSV template."""
    sample_csv_text = (
        "Student_ID,Age,Gender,Family_Income,Internet_Access,Study_Hours_per_Day,Attendance_Rate,Assignment_Delay_Days,Travel_Time_Minutes,Part_Time_Job,Scholarship,Stress_Index,GPA,Semester_GPA,CGPA,Semester,Department\n"
        "1001,20.5,Male,35000,Yes,5.0,92.0,0,25,No,Yes,3.5,3.40,3.50,3.45,Year 2,CS\n"
        "1002,22.0,Female,28000,No,2.0,65.0,4,55,Yes,No,8.0,1.80,1.60,1.70,Year 3,Arts\n"
        "1003,19.8,Male,42000,Yes,4.0,84.5,1,30,No,No,5.0,2.65,2.70,2.68,Year 1,Engineering\n"
        "1004,21.2,Female,31000,Yes,3.0,72.0,3,40,Yes,No,7.2,2.10,1.95,2.05,Year 2,Business\n"
        "1005,23.1,Male,55000,Yes,6.5,95.0,0,15,No,Yes,2.8,3.85,3.90,3.88,Year 4,Science\n"
    )
    return send_file(
        io.BytesIO(sample_csv_text.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_student_dataset.csv'
    )    


if __name__ == '__main__':
    app.run(port=5000, debug=True)