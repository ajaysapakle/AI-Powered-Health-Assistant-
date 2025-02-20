import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
from sklearn.model_selection import train_test_split
import random
import warnings

#from Clap import MainClapExe
#MainClapExe()

warnings.simplefilter('ignore')

# nltk.download("punkt")

def speak(text):
    engine = pyttsx3.init()
    Id = r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'
    engine.setProperty('voice',Id)
    print("")
    print(f"==> AI Health Assistant: {text}")
    print("")
    engine.say(text=text)
    engine.runAndWait()

 

def speechrecognition():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening......")
        r.pause_threshold = 1
        audio = r.listen(source,0,1)

    try:
        print("Recogizing.....") 
        query = r.recognize_google(audio,language="en")   
        return query.lower()

    except:
        return ""    

    speechrecognition()   

def MainExecution(query):
    Query = str(query).lower()

    if "hello"  in Query:
        speak("Hello sir, Wellcome Back!")

    elif "bye"  in Query:
        speak("Nice to meet you sir, Have nice day!")

    elif "time"  in Query:
        from datetime import datetime
        time = datetime.now().strftime("%H:%M")
        speak(f" The time is Now Is :{time}")

intents = {

        "health_greetings": {
            "patterns": ["hello doctor", "hi doctor", "good morning doctor", "good afternoon doctor", "good evening doctor", "doctor help", "doctor, I need help", "doc, I'm not feeling good"],
            "responses": ["Hello, how can I assist you with your health today?", "Good morning/afternoon/evening. Please tell me about your concerns.", "Hello, I'm here to help. What brings you in today?"],
            "context": ["general"]
        },
        "health_goodbye": {
            "patterns": ["thank you doctor", "bye doctor", "goodbye doctor", "see you later doctor", "farewell doctor", "thanks for your help doctor", "I'm going now doctor"],
            "responses": ["You're welcome. Take care and feel better.", "Goodbye. Don't hesitate to reach out if you need further assistance.", "Take care. See you next time."],
            "context": ["general"]
        },
        "symptom_check": {
            "patterns": ["I'm feeling [symptom]", "I have [symptom]", "I'm experiencing [symptom]", "my [body part] hurts", "I've been having [symptom] for [duration]", "I'm not feeling well", "I've got a [symptom]", "I've got [symptom] for some time now"],
            "responses": [
                "Please describe your symptoms in detail. When did they start? Are there any other associated symptoms?",
                "Tell me more about how you're feeling. Where exactly is the pain? How severe is it?",
                "Let's go through your symptoms. What makes them better or worse? Any changes in your routine?"
            ],
            "context": ["symptoms"]
        },
        "medical_history": {
            "patterns": ["I have a history of [condition]", "I'm taking [medication]", "I've been diagnosed with [condition]", "I'm allergic to [allergy]", "I've had [condition] before", "I'm on [medication]", "I've got this before"],
            "responses": [
                "Thank you for that information. It's important to know your medical history. Please tell me more about your current medications and dosages.",
                "Knowing your medical history helps me understand your situation better. Are there any recent changes in your health?",
                "Please provide a list of all your current medications, including over-the-counter drugs and supplements."
            ],
            "context": ["medical_history"]
        },
        "diagnosis_request": {
            "patterns": ["what do you think is wrong with me?", "what could this be?", "what's the diagnosis?", "what illness do I have?", "what's causing this?", "am I sick?", "what's happening to me?"],
            "responses": [
                "Based on your symptoms, it could be a few things. However, I cannot provide a definitive diagnosis. It's important to get a proper evaluation from a medical professional.",
                "It's difficult to say without a physical examination and further tests. I recommend you schedule an appointment with a doctor for a thorough check-up.",
                "I can't give you a diagnosis, but your symptoms suggest you should seek in-person medical advice."
            ],
            "context": ["diagnosis"]
        },
        "medication_request": {
            "patterns": ["what medicine should I take?", "can you prescribe something?", "what can I take for [symptom]?", "what's good for [symptom]?", "any medicine for this?", "what pill do I need?"],
            "responses": [
                "I cannot prescribe medication. Prescriptions must be issued by a licensed medical doctor after a proper examination.",
                "For medication recommendations, please consult a pharmacist or doctor. They can assess your specific needs and provide appropriate treatment.",
                "I can give general advice on over-the-counter remedies, but I cannot prescribe anything. Please see a doctor for prescription medication."
            ],
            "context": ["medication"]
        },
        "lifestyle_tips": {
            "patterns": ["how can I improve my health?", "what lifestyle changes should I make?", "any health tips?", "how to stay healthy?", "what should I do?", "how to get better?"],
            "responses": [
                "Maintaining a balanced diet, regular exercise, and adequate sleep are crucial for overall health. I can provide more specific tips based on your lifestyle.",
                "Reducing stress, staying hydrated, and avoiding excessive alcohol and tobacco use are important. I can suggest stress management techniques if needed.",
                "Regular check-ups, vaccinations, and preventive care are essential. Let’s discuss your current health habits and how to make improvements."
            ],
            "context": ["lifestyle"]
        },
        "specific_tips": {
            "patterns": ["what should I eat for [condition]?", "how to exercise with [condition]?", "how to manage stress due to [condition]?", "how to control [condition]?", "treatment for [condition]?", "how to deal with [condition]?"],
            "responses": [
                "For [condition], it's important to focus on [dietary advice]. Remember to consult with a nutritionist for a personalized plan.",
                "When exercising with [condition], it's crucial to [exercise advice]. Always check with your doctor before starting a new exercise regimen.",
                "Managing stress with [condition] can be challenging. Try [stress management techniques]. Consider seeking support from a therapist or counselor.",
                "To control [condition], [general advice]. Always consult a doctor for a full treatment plan",
                "Treatment for [condition] generally involves [treatment overview]. Please see a doctor for specific treatment."
            ],
            "context": ["specific_tips"]
        },
        "referral_request": {
            "patterns": ["can you recommend a specialist?", "where can I find a doctor for [condition]?", "I need to see a specialist", "who should I see for this?", "who can help me with this?"],
            "responses": [
                "I can help you find a specialist in your area. Please provide your location, and I'll look for doctors specializing in [condition].",
                "For [condition], I recommend seeing a [specialist type]. I can provide a list of specialists near you.",
                "I can refer you to a specialist for further evaluation. Please provide your location and insurance details."
            ],
            "context": ["referral"]
        },
        "emergency": {
            "patterns": ["I'm having a medical emergency", "call emergency services", "I need immediate help", "help me now!", "this is an emergency", "I'm dying"],
            "responses": [
                "If you are experiencing a medical emergency, please call your local emergency services immediately.",
                "For immediate medical assistance, dial your emergency number now.",
                "I cannot provide emergency medical assistance. Please contact emergency services right away."
            ],
            "context": ["emergency"]
        },
        "disease_control": {
            "patterns": ["how to control [disease]?", "what are the ways to manage [disease]?", "how to live with [disease]?", "ways to reduce [disease] symptoms?", "how to keep [disease] in check?"],
            "responses": [
                "Controlling [disease] usually involves [lifestyle changes, medication, therapy, etc.]. Please consult your doctor for personalized advice.",
                "Managing [disease] can be done through [specific strategies]. Always remember to follow your doctor's recommendations.",
                "Living with [disease] requires [daily management tips]. Regular check-ups are essential.",
                "Reducing symptoms of [disease] can be achieved by [specific techniques]. Please consult a healthcare professional for guidance."
            ],
            "context": ["disease_control"]
        },
        "disease_treatment": {
            "patterns": ["what is the treatment for [disease]?", "how is [disease] treated?", "what are the treatment options for [disease]?", "is there a cure for [disease]?", "how to fix [disease]?"],
            "responses": [
                "Treatment for [disease] typically includes [medical procedures, therapies, medications]. Consult your doctor for the best course of action.",
                "[disease] is treated by [specific treatments]. Always seek professional medical advice.",
                "Treatment options for [disease] vary depending on the severity and individual factors. Discuss your options with your doctor.",
                "Whether or not there is a cure for [disease] depends on the specific condition. Please consult your doctor for accurate information."
            ],
            "context": ["disease_treatment"]
        }
    }

training_data = []
labels = []

for intent , data in intents.items():
    for pattern in data['patterns']:
        training_data.append(pattern.lower())
        labels.append(intent)

# print(training_data)
# print(labels)        


Vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize,stop_words="english",max_df=0.8,min_df=1)
X_train = Vectorizer.fit_transform(training_data)
X_train,X_test,Y_train,Y_test = train_test_split(X_train,labels,test_size=0.4,random_state=42,stratify=labels)

model = SVC(kernel='linear', probability=True, C=1.0)
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

def predict_intent(user_input):
    user_input = user_input.lower()
    input_vector = Vectorizer.transform([user_input])
    intent = model.predict(input_vector)[0]
    return intent

print("AI Assistant: Hello! How can I help you?")
while True:
    user_input = speechrecognition()
    if user_input.lower() == 'exit':
        print("AI Assistant: Goodbye!")
        break

    intent = predict_intent(user_input)
    if intent in intents:
        responses = intents[intent]['responses']
        response = random.choice(responses)
        speak(response)

    else:
        speak("AI Assistant: Sorry, I'm not sure how to respond to that.")