import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example: User provides a document
document = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans using natural language. It involves the development of algorithms and models to understand, interpret, and generate human-like text. NLP is used in various applications, including chatbots, sentiment analysis, and language translation.
"""

# Preprocess the document
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Tokenize and preprocess the document
corpus = [preprocess_text(document)]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Function to get the most relevant response based on user input
def get_response(user_input):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between the user input and the document
    similarity_scores = cosine_similarity(user_vector, X)
    most_similar_index = similarity_scores.argmax()

    # Get the most relevant response
    response = corpus[most_similar_index]

    return response

# User interaction
print("Document:")
print(document)

while True:
    user_query = input("Ask me a question (type 'quit' to exit): ")
    if user_query.lower() == 'quit':
        print("Goodbye!")
        break

    response = get_response(user_query)
    print("Response:")
    print(response)
