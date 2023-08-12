import openai
import pyttsx3
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import datetime
from APIkey import openaiKeyCode

openai.api_key = openaiKeyCode
openai.api_version = None

completion = openai.ChatCompletion.create

def Reply(question):
    prompt = f'Tony Stark:{question}\n Jarvis: '
    response = completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are Jarvis, Tony Stark's AI and I am Tony Stark. Act like that for the following conversation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer

engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

speak("Hello How Are You? ")

def takeCommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening....')
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing.....")
        query=r.recognize_google(audio, language='en-in')
        print("Tony Stark Said: {} \n".format(query))
    except Exception as e:
        print("Say That Again....")
        return "None"
    return query


if __name__ == '__main__':
    while True:
        query=takeCommand().lower()

        if 'tools' in query:
            if 'wikipedia' in query:
                speak('Searching Wikipedia...')
                query = query.replace("wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)
            elif 'open youtube' in query:
                webbrowser.open("youtube.com")
            elif 'open google' in query:
                webbrowser.open("google.com")
            elif 'open howdy' in query:
                webbrowser.open("https://howdy.tamu.edu/uPortal/f/welcome/normal/render.uP")

            elif 'the time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                speak(f"Sir, the time is {strTime}")

            elif 'open code' in query:
                codePath = "C:\\Users\\nisha\\PycharmProjects\\pythonProject\\Jarvis.py"
                os.startfile(codePath)

        else:
            ans = Reply(query)
            print(ans)
            speak(ans)

        if 'shut down' or 'quit' or 'bye' in query:
            break