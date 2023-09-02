
from flask import Flask,redirect,url_for,render_template,request
from twilio.rest import Client

from chat import get_response


#for emotion analysis
import joblib
loaded_model = joblib.load('emotion_model.joblib')
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

# flask part 
app=Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['msg']
        predicted_label = loaded_model.predict([user_input])
      
        got_num=0
        num=user_input[0:13] #number on which message needs to be sent.
        num_without_plus=user_input[1:13]
        user_name=user_input[14:]
        print(num)
        print(user_name)
        if(num_without_plus.isnumeric()): #to check the validity of number if provided
          got_num=1; 
          emergency_msg=f'your friend {user_name} seems to be in trouble. Please reach out to him -- hearky'
          
          account_sid = 'ACfb7b5de7cd4934430469c62b5ebf98cc'
          auth_token = '1ab7c5ceb23fb3a4655047a399877348'
          client = Client(account_sid, auth_token)
          message = client.messages.create(
            from_='+16184378302',
            body=emergency_msg,
            to=num
          )
          if(message.sid):
            print("message sent")

          return render_template('emergency.html')

        else:
          text=get_response(user_input)
          print(predicted_label)
          return render_template('msg.html',botResponse=text,emotion_num=predicted_label)
    return render_template('index.html')
       

if __name__=="__main__":
  app.run(host='0.0.0.0', port=8000)
