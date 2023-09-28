from flask import Flask, request, jsonify
from torch import cuda, bfloat16
import transformers
import requests
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from transformers import StoppingCriteria, StoppingCriteriaList
import os
import torch
from langchain.llms import HuggingFacePipeline

from flask import Flask, request, jsonify, render_template

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os

app = Flask(__name__)

# Define your model initialization code here
model_id = 'meta-llama/Llama-2-7b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# ... (Rest of your model initialization code)
# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Can you please let us know more details about your ",
})
# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)


stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

# import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids


# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


llm = HuggingFacePipeline(pipeline=generate_text)


#######
path = "CSV_FiLES"  # give path of csv files


documents = []
for file in os.listdir(path):
    if file.endswith('.csv'):
        csv_path = path+'/' + file
        loader = CSVLoader(csv_path)
        documents.extend(loader.load_and_split())
loader1 = TextLoader("Text files", encoding = "utf-8")
documents.extend(loader1.load_and_split())
print(documents)



# from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)


# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

# from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)


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
    preprocessed_question = preprocess_question(question)
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

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/api/answer', methods=['POST'])
def get_answer():
    try:
        # Get the user question from the JSON request
        data = request.get_json()
        user_question__ = data.get('user_question')

        # Call your Answer__ function to get the answer
        answer = Answer__(user_question__)

        # Return the answer as JSON response
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
