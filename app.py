from flask import Flask, render_template, request
import prediction

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        frenchtext=prediction.start_prediction(request.form['englishtext'])
        return render_template('index.html', message=frenchtext)
    return render_template('index.html')

if __name__== '__main__':
    app.run(debug=True)