!pip install transformers==4.29.2
!pip install accelerate==0.19.0
!pip install torch==2.0.0
!pip install einops==0.6.1
!pip install flask
!pip install pytelegrambotapi
!pip install python-dotenv
!pip install requests

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


import telebot
bot=telebot.TeleBot('6447345913:**********************')


@bot.message_handler(commands=['ai','info'])
def ai_handler(message):
  if message.text=='/ai':
    bot.reply_to(message,'Be clear pal. What do you want talk about? 🙂 ')
  if message.text=='/info':
    bot.reply_to(message,'Sup!!🙂 i am hearky , your Ai telebot created by Master Atulya.')
    bot.reply_to(message,'Ask me anything (use /ai followed by your text), will try my best to answer it ✊')
  else:
    refine_message=message.text.replace('/ai','')
    emotion_num=loaded_model.predict([refine_message]) #predicts the sentiment number
    
    response=f"""
    {get_response(refine_message)}
    """
    senti_msg=''
    if(emotion_num==0):
      senti_msg="you seem angry. I know the whole world sucks sometimes,but we gotta keep our cool and move on."
  
    elif(emotion_num==2):
      senti_msg="I have texted the authorities and your family memebers, until they reach out to you just keep calm and keep it low"  
  
    elif (emotion_num==6):
      senti_msg="We all have our lows and highs. These times test us and we need to keep our hearts strong and face it with all of our might.Just believe and everything's gonna be fine ✊"  

    elif (emotion_num==7 or emotion_num==3):
      senti_msg="There are times when we regret the choices we make, but even then we have our options, whether to keep on thinking about the mistakes or take a choice to somehow mend it. The choices we make at this point makes all the difference my friend :"
    
    bot.reply_to(message,response)
    bot.reply_to(message,senti_msg)

if __name__=='__main__':
  print('bot is running')
  bot.polling()


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
