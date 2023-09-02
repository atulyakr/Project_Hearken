
from flask import Flask,redirect,url_for,render_template,request
from twilio.rest import Client

# model call
import torch
from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2",
    use_fast=False,
    padding_side="left",
    trust_remote_code=True,
)

generate_text = pipeline(
    model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_fast=False,
    device_map={"": "cuda:0"},
)

def get_response(msg):
  res = generate_text(
    msg,
    min_new_tokens=2,
    max_new_tokens=100,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True)
  return res[0]["generated_text"]


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
          auth_token = 'e67df05c397e2851b7c06c5e4c7e0ec2'
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
  app.run(debug=False,host='0.0.0.0')
