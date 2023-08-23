!pip install transformers==4.29.2
!pip install accelerate==0.19.0
!pip install torch==2.0.0
!pip install einops==0.6.1
!pip install flask

from flask import Flask,redirect,url_for,render_template,request
from chat import get_response

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))


app=Flask(__name__,template_folder='/content/templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = request.form['msg']
        text=get_response(input_data)
        return render_template('msg.html',botResponse=text)
    return render_template('index.html')

if __name__=="__main__":
  app.run()

