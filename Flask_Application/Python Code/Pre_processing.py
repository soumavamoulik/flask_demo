#<<<<<<<<<<<<<<<<<<<<<<<<<< CONDITIONS FOR PREPROCESSING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

chat_history = []

def preprocess_document(document):
    # Tokenization and cleaning
    sentences = nltk.sent_tokenize(document.lower())  # Convert to lowercase and split into sentences
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]  # Tokenize each sentence

    # Stopword removal
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [
        [word for word in sentence_tokens if word not in stopwords_set]
        for sentence_tokens in tokens
    ]

    # Lemmatization (or stemming if preferred)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        [lemmatizer.lemmatize(word) for word in sentence_tokens]
        for sentence_tokens in filtered_tokens
    ]

    return lemmatized_tokens


# Load documents from a text file
document_file_path = ' '  # path to your document file

with open(document_file_path, 'r', encoding='utf-8') as file:
    document = file.readlines()


preprocessed_documents = []

for doc in document:
    preprocessed_doc = preprocess_document(doc)

    # Create a set to store unique tokens for  document
    unique_tokens = set()

    # Remove duplicate tokens and add the remaining tokens to the preprocessed document
    for sentence_tokens in preprocessed_doc:
        unique_tokens.update(sentence_tokens)

    # Convert the set back to a list
    preprocessed_doc = [list(unique_tokens)]

    preprocessed_documents.append(preprocessed_doc)

# print(f"preprocessed_documents-->{preprocessed_documents}")


# Flatten the preprocessed document list
flat_document_tokens = [token for sentence_tokens in preprocessed_documents for token in sentence_tokens[0]]

# Define a list of unnecessary words to be removed (customize this list as needed)
unnecessary_words = ["the", "and", "in", "to", "is", "it", "of", "for", "a", "on", "with"]

# Filter out unnecessary words
filtered_document_tokens = [token for token in flat_document_tokens if token not in unnecessary_words]

# Extract unique keywords
unique_keywords = list(set(filtered_document_tokens))


# Define a function to remove special characters and single characters
def is_valid_keyword(keyword):
    return len(keyword) > 1 and not all(char in string.punctuation for char in keyword)

# Filter out special characters and single characters
filtered_keywords = [keyword for keyword in unique_keywords if is_valid_keyword(keyword)]

# Define a regular expression pattern to match keywords containing only letters
pattern = re.compile(r'^[a-zA-Z]+$')

# Filter out keywords that match the pattern (contain only letters)
final_filtered_keywords = [keyword for keyword in filtered_keywords if pattern.match(keyword)]

# Print the final list of filtered keywords
# print(f"final filtered keywords-->{final_filtered_keywords}")

def preprocess_question(question):
    # Tokenization and cleaning
    tokens = nltk.word_tokenize(question.lower())  # Convert to lowercase

    # Stopword removal
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stopwords_set]

    # Lemmatization (or stemming if preferred)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return lemmatized_tokens


# Function to preprocess a text document
def preprocess_document_(doc):
    # Preprocess the document using the same steps as question preprocessing
    tokens = nltk.word_tokenize(doc.lower())
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stopwords_set]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)



# Preprocess documents
preprocessed_documents_ = [preprocess_document_(doc) for doc in document]

# Function to preprocess a question with conditions
def preprocess_question_with_conditions(question, final_keywords, similarity_threshold, documents):
    # Tokenization and cleaning
    tokens = nltk.word_tokenize(question.lower())  # Convert to lowercase

    # Stopword removal using final keywords as stopwords
    # stopwords_set = set(final_keywords)
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stopwords_set]

    # Lemmatization (or stemming if preferred)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Check if some tokens in the question match final keywords
    matching_tokens = [token for token in lemmatized_tokens if token in final_filtered_keywords]

    # Preprocess user question question
    preprocessed_question = preprocess_question(user_question)
    # preprocessed_question = lemmatized_tokens

    # Calculate TF-IDF similarity with documents
    vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform([question] + documents)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents_)
    user_question_vector = vectorizer.transform([' '.join(preprocessed_question)])

    # Calculate cosine similarities between the user question and documents
    similarities = cosine_similarity(user_question_vector, tfidf_matrix)

    # similarities = cosine_similarity(user_question_vector, tfidf_matrix[1:])
    top_similar_documents_indices = similarities.argsort()[0][::-1][:5]
    top_similar_documents = [documents[idx] for idx in top_similar_documents_indices]

    # Check if any document is above the similarity threshold
    document_matches = [(i,doc) for i,doc in enumerate(top_similar_documents, start=1) if similarities[0, top_similar_documents_indices[0]] > similarity_threshold]

    return matching_tokens, document_matches



# Define your question answering code as a function
def Answer__(user_question_):
    # Your existing code here
        # user_question = "What are the different volume indicators for BP products? "
    user_question = user_question_
    similarity_threshold = 0.2
    matching_tokens, document_matches = preprocess_question_with_conditions(user_question, final_filtered_keywords, similarity_threshold, preprocessed_documents_)

    # Check conditions
    if matching_tokens and document_matches:
        print("Condition 1: Matching tokens found in the final keywords.")
        # print("Matching tokens:", matching_tokens)
        print("Condition 2: Similar documents found based on TF-IDF similarity.")
        # print("Matching document indices:", document_matches)
        query = user_question
        result = chain({"question": query, "chat_history": chat_history })
        answer = result['answer']
        model_answer = answer



    elif document_matches:
        print("Condition 2: Similar documents found based on TF-IDF similarity.")
        # print("Matching document indices:", document_matches)
        query = user_question
        result = chain({"question": query, "chat_history": chat_history })
        answer = result['answer']
        model_answer = answer
        


    elif matching_tokens:
        print("Condition 1: Matching tokens found in the final keywords.")
        # print("Matching tokens:", matching_tokens)
        query = user_question
        result = chain({"question": query, "chat_history": chat_history })
        answer = result['answer']
        model_answer = answer
        


    else:
        print("No condition passed.")
        answer = "No condition passed."
        model_answer = answer
    
    return model_answer
