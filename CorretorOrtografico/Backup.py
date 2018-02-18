# Python
# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))


import speech_recognition as sr
r = sr.Recognizer()

print(sr.Microphone.list_microphone_names())

with sr.Microphone() as source:                # use the default microphone as the audio source
    audio = r.listen(source)                   # listen for the first phrase and extract it into audio data

try:
    print("You said " + r.recognize_google(audio))    # recognize speech using Google Speech Recognition
except LookupError:                            # speech is unintelligible
    print("Could not understand audio")

if(False):


    import speech_recognition as sr

    r = sr.Recognizer()
    with sr.WavFile("test.wav") as source:  # use "test.wav" as the audio source
        audio = r.record(source)  # extract audio data from the file

    try:
        print("Transcription: " + r.recognize_google(audio))  # recognize speech using Google Speech Recognition
    except LookupError:  # speech is unintelligible
        print("Could not understand audio")


