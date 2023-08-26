!pip install transformers==4.29.2
!pip install accelerate==0.19.0
!pip install torch==2.0.0
!pip install einops==0.6.1
!pip install flask

from flask import Flask,redirect,url_for,render_template,request
from chat import get_response

#for emotion analysis
import joblib
loaded_model = joblib.load('/content/emotion_model.joblib')
#sentiment scores are as followed
'''Anger-0
  Disgust-1
  Fear-2
  Guilty-3
  Joy-4
  Love-5
  Sadness-6
  Shame-7
  surprise-8'''

#a public url will be generated on which our web app will be rendered locally(when using google colab)

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))


app=Flask(__name__,template_folder='/content/templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['msg']
        text=get_response(user_input)
        predicted_label = loaded_model.predict([user_input])
        return render_template('msg.html',botResponse=text,emotion_num=predicted_label)
    return render_template('index.html')

if __name__=="__main__":
  app.run()
