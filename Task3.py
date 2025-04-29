# AI CHATBOT WITH NLP Using Python.
# BUILD A CHATBOT USING NATURAL LANGUAGE PROCESSING LIBRARIES LIKE NLTK OR SPACY, CAPABLE OF ANSWERING USER QUERIES.
# DELIVERABLE: A PYTHON SCRIPT AND A WORKING CHATBOT.
import nltk
import spacy
import random
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.chat.util import Chat, reflections
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk.downloader
import string

from dotenv import load_dotenv
import os

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define patterns and responses
pairs = [
    [
        r"hi|hello|hey|greetings",
        ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Greetings! How may I help you?"]
    ],
    [
        r"how are you|how's it going",
        ["I'm just a chatbot, but I'm functioning well! How about you?", "I don't have feelings, but thanks for asking! How can I help you?"]
    ],
    [
        r"what is your name|who are you",
        ["I'm an NLP-powered chatbot. You can call me ChatBot.", "I'm your friendly neighborhood chatbot!"]
    ],
    [
        r"bye|goodbye|see you later",
        ["Goodbye! Have a great day!", "See you later! Come back if you have more questions."]
    ],
    [
        r"thank you|thanks",
        ["You're welcome!", "No problem! Happy to help.", "Anytime!"]
    ],
    [
        r"(.*) (weather|temperature) (.*)",
        ["I'm sorry, I don't have real-time weather data. You might want to check a weather website or app."]
    ],
    [
        r"(.*) (age|old) (.*)",
        ["I'm a chatbot, so I don't have an age. I was just created recently!"]
    ],
    [
        r"(.*) (help|support|assistance) (.*)",
        ["I can help with general questions. Try asking me about common topics or say 'help' to see options."]
    ],
    [
        r"(.*) (time|date) (.*)",
        ["I don't have access to real-time clock data. Check your device's clock for the current time."]
    ],
    [
        r"what can you do|help",
        ["I can answer general questions, have simple conversations, and provide information on predefined topics. Try asking me something!"]
    ],
    [
        r"(.*)",
        ["I'm not sure I understand. Could you rephrase that?", "I'm still learning. Can you ask me something else?", "That's interesting. Tell me more."]
    ]
]

class NLPChatBot:
    def __init__(self):
        self.chatbot = Chat(pairs, reflections)
        self.context = {}
        
    def preprocess_text(self, text):
        # Tokenize and lemmatize
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        return ' '.join(tokens)
    
    def extract_entities(self, text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def respond(self, user_input):
        # Preprocess the input
        processed_input = self.preprocess_text(user_input)
        
        # Extract named entities for potential context
        entities = self.extract_entities(user_input)
        if entities:
            self.context['entities'] = entities
        
        # Get response from the chatbot
        response = self.chatbot.respond(processed_input)
        
        # If no matching pattern was found, use a fallback with NLP analysis
        if not response:
            doc = nlp(user_input)
            
            # Check for question types
            if any(token.text.lower() in ['who', 'what', 'when', 'where', 'why', 'how'] for token in doc):
                response = "That's an interesting question. I might not have the full answer, but I can try to help."
            elif any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in doc):
                action = [token.lemma_ for token in doc if token.dep_ == 'ROOT' and token.pos_ == 'VERB'][0]
                response = f"I understand you want to {action}. Can you provide more details?"
            else:
                response = "I'm still learning. Could you rephrase that or ask me something else?"
        
        return response

def main():
    print("NLP ChatBot: Hello! I'm a chatbot with some NLP capabilities. Type 'bye' to exit.")
    chatbot = NLPChatBot()
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("NLP ChatBot: Goodbye! Have a great day!")
                break
            
            response = chatbot.respond(user_input)
            print("NLP ChatBot:", response)
            
        except KeyboardInterrupt:
            print("\nNLP ChatBot: Goodbye!")
            break
        except Exception as e:
            print("NLP ChatBot: Sorry, I encountered an error. Let's try again.")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()