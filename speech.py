import pyttsx3

def say(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 130)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    say("Wie geht es dir, Dawid?")