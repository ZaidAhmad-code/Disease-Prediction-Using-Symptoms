from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and encoders
model = joblib.load('model/disease_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
symptom_list = joblib.load('model/symptom_list.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        input_vector = np.zeros(len(symptom_list))

        for symptom in selected_symptoms:
            if symptom in symptom_list:
                index = np.where(symptom_list == symptom)[0][0]
                input_vector[index] = 1

        prediction = model.predict([input_vector])[0]
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        return render_template('index.html', symptoms=symptom_list, prediction=predicted_disease)

    return render_template('index.html', symptoms=symptom_list, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
