#Import relevant libraries
import os  # System-dependent functionality like reading or writing
import aiofiles  # Asynchronous file operations
import asyncio   # Support for asynchronous programming
import nest_asyncio  # Allows nested use of asyncio event loops
from sentence_transformers import SentenceTransformer # Pre-trained models for transforming sentences into dense vector embeddings
import faiss  # Similarity search and clustering of dense vectors
import numpy as np  # Handle large, multi-dimensional arrays and matrices
from transformers import AutoModelForCausalLM, AutoTokenizer  # Pre-trained models and tokenizers for natural language processing tasks
import torch  # Import torch for CUDA support
import re  # Support for regular expressions
from bs4 import BeautifulSoup  # Parsin HTML document
import logging   # Creates logging messages to help debbug
from nltk.corpus import wordnet # Import WordNet lexical database for synonym Lookup
import nltk # Tools for working with human language
from rouge_score import rouge_scorer  # Import ROUGE scorer
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity  # Import Cosine Similarity
from config import FILE_PATH   # Load the file path from the  config.py file
# Ensure necessary NLTK data is downloaded
nltk.download('wordnet')

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Note loggings will not be commented


def preprocess_markdown(content):
    """
    Preprocess Markdown content to clean and structure it.
    This includes converting Markdown to plain text, removing headers, 
    and formatting HTTPS links.
    """

    logging.info("Preprocessing markdown content") 

    # Inner function to format 'https' links in the text
    def format_https_links(text):
        # Define a regular expression pattern to match 'https' links
        pattern = r'https://[^\s"\']+'
        # Replace matched links with a formatted reference
        return re.sub(pattern, lambda x: f'[Reference: {x.group()}]', text)
    
    # Apply the inner function to format 'https' links in the markdown content
    content = format_https_links(content)

    # Remove leading hash symbols from markdown headings
    content = re.sub(r'^(#{1,6})\s+', '', content, flags=re.MULTILINE)
    
    # Parse the markdown content as HTML using BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract plain text from the HTML
    preprocessed_text = soup.get_text()

    logging.info("Finished preprocessing markdown content")

    # Return the cleaned and structured plain text
    return preprocessed_text

#################################################################
### Step1 : Load and chunk the Markdown files asynchronously ####
#################################################################

async def process_md_file(file_path, chunk_size=512):
    """
    Asynchronously read and preprocess a Markdown file, then chunk the content.
    """
    logging.info(f"Processing file: {file_path}")
    
    # Open the markdown file asynchronously in read mode with UTF-8 encoding
    async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
        # Read the entire content of the file asynchronously
        content = await file.read()

    # Preprocess the content to clean and structure it
    content = preprocess_markdown(content)
    
    # Tokenize the content into words and chunk it into specified sizes
    tokens = content.split()    # Split content into individual words (tokens)
    # Create chunks of tokens with the specified size
    chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    
    # Log an informational message indicating the number of chunks created
    logging.info(f"File processed and chunked into {len(chunks)} chunks")

     # Return the list of chunks
    return chunks

async def process_files(folder_path, chunk_size=512):
    """
    Asynchronously process all Markdown files in a folder and return their chunks.
    """
    logging.info(f"Processing files in folder: {folder_path}")
    
    all_chunks = [] # List to store chunks from all files
    tasks = [] # List to hold asynchronous tasks for processing files

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file has a '.md' extension
        if filename.endswith('.md'):
            # Construct the full path to the markdown file
            file_path = os.path.join(folder_path, filename)
            # Add a task to process the markdown file asynchronously
            tasks.append(process_md_file(file_path, chunk_size))
    
    # Gather results from all asynchronous tasks
    results = await asyncio.gather(*tasks)

    # Flatten the list of chunks from all files into a single list
    for chunks in results:
        all_chunks.extend(chunks)

    logging.info(f"All files processed. Total chunks: {len(all_chunks)}")

    # Return the combined list of chunks
    return all_chunks


###################################
### Step2 :  Perform Embedding ####
###################################
# Instantiate a SentenceTransformer model with the 'all-MiniLM-L6-v2' architecture
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    """
    Generate embeddings for a list of text chunks using the SentenceTransformer model.
    """
    logging.info("Generating embeddings")

    # Generate embeddings for the list of text chunks using the SentenceTransformer model
    # `show_progress_bar=True` displays a progress bar during encoding
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    logging.info("Embeddings generated")

    # Return the generated embeddings
    return embeddings


#######################################
### Step 3: Create the FAISS index ####
#######################################
def create_faiss_index(embeddings):
    """
    Create a FAISS index from the given embeddings.
    """
    logging.info("Creating FAISS index")
    
    # Convert the list of embeddings to a NumPy array of type float32
    embeddings_np = np.array(embeddings).astype(np.float32)

    # Get the dimensionality of the embeddings (number of features)
    dim = embeddings_np.shape[1]

    # Create a FAISS index for L2 (Euclidean) distance
    index = faiss.IndexFlatL2(dim)

    # Add embeddings to the FAISS index
    index.add(embeddings_np)
    
    logging.info("FAISS index created")

    # Return the created FAISS index
    return index


def search_index(index, query_embeddings, k=5):
    """
    Search the FAISS index for the top k nearest neighbors of the query embeddings.
    """
    logging.info(f"Searching index for top {k} results")

    # Perform the search on the FAISS index with the query embeddings
    # Returns distances and indices of the nearest neighbors
    distances, indices = index.search(query_embeddings, k)

    logging.info("Search completed")
    
    # Log the completion of the search
    return distances, indices


def rebuild_index(folder_path, chunk_size=512):
    """
    Rebuild the FAISS index by processing files in the specified folder and generating embeddings.
    """
    logging.info("Rebuilding FAISS index")

    # Process all markdown files in the specified folder, chunking the content
    chunks = asyncio.run(process_files(folder_path, chunk_size))

    # Generate embeddings for the processed chunks
    embeddings = generate_embeddings(chunks)

    # Get the dimensionality of the embeddings
    dim = embeddings.shape[1]

    # Create a new FAISS index for L2 (Euclidean) distance
    new_index = faiss.IndexFlatL2(dim)

    # Add the new embeddings to the new FAISS index
    new_index.add(np.array(embeddings).astype(np.float32))
    
    logging.info("FAISS index rebuilt successfully")
    # Return both the new FAISS index and the processed chunk
    return new_index, chunks  # Return both index and chunks


##################################################
### # Step 4: Implement text generation model  ###
##################################################
# Define the model name for GPT-Neo with 1.3 billion parameters
model_name = "EleutherAI/gpt-neo-1.3B"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to the eos_token if pad_token is not already set
# This ensures that padding tokens are handled consistently
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the GPT-Neo model for causal language modeling
gpt_model = AutoModelForCausalLM.from_pretrained(model_name)

# Determine whether to use GPU or CPU based on availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move the GPT model to the selected device
gpt_model.to(device)  

# Define a few-shot prompt with examples to guide the model's responses
few_shot_prompt = """
You are a helpful assistant that answers developers' questions by providing information from the relevant documentation. Always include the reference link at the end of your answer.

Example 1:
Question: What are SageMaker Geospatial capabilities?
Answer: SageMaker Geospatial capabilities allow developers to process and analyze geospatial data, such as satellite imagery, within the SageMaker environment. This includes operations like data ingestion, preprocessing, and model training on geospatial data. [Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/geospatial-overview.html]

Example 2:
Question: How do I create an EC2 instance using the AWS CLI?
Answer: To create an EC2 instance using the AWS CLI, you can use the aws ec2 run-instances command, specifying the necessary parameters like --image-id, --instance-type, and --key-name. [Reference: https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html]

Example 3:
Question: What is an AWS IAM role?
Answer: An AWS IAM role is a set of permissions that define what actions an entity (such as an EC2 instance or an AWS Lambda function) can perform in the AWS environment. Roles are used to grant permissions to AWS resources and services without needing to share long-term credentials. [Reference: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html]

Now, answer the following question:
Question: {question}
Answer:
"""


def get_synonyms(word):
    """
    Retrieve synonyms for a given word using WordNet.
    """
    # Initialize an empty set to store synonyms
    synonyms = set()

    # Retrieve synsets (sets of synonyms) for the given word
    for syn in wordnet.synsets(word):
        # Iterate through each lemma in the synset
        for lemma in syn.lemmas():
            # Add the lemma name (synonym) to the set
            synonyms.add(lemma.name())

    # Return the list of synonyms
    return list(synonyms)

def handle_oov_terms(question, known_terms):
    """
    Handle out-of-vocabulary terms in the question by replacing them with synonyms.
    """
    # Split the question into individual words
    words = question.split()
    
    # Initialize an empty list to store the updated words
    new_words = []

    # Iterate through each word in the question
    for word in words:
        # Check if the word is not in the known terms
        if word.lower() not in known_terms:
            # Retrieve synonyms for the out-of-vocabulary (OOV) word
            synonyms = get_synonyms(word)
            # If synonyms are found, use the first synonym as a fallback
            if synonyms:
                new_words.append(synonyms[0])  
            else:
                # If no synonyms are found, keep the original word
                new_words.append(word)
        else:
            # If the word is in the known terms, keep it unchanged
            new_words.append(word)
    
    # Join the updated words into a single string and return it
    return ' '.join(new_words)


def generate_answer(question, relevant_docs, known_terms):
    """
    Generate an answer to the question using the GPT model and relevant documentation.
    """
    logging.info(f"Generating answer for question: {question}")

    # Handle Out-of-Vocabulary (OOV) terms in the question by replacing them with synonyms
    question = handle_oov_terms(question, known_terms)
    
    # Format the prompt with the question and relevant documents
    prompt = few_shot_prompt.format(question=question)
    for doc in relevant_docs:
        # Append each document's content and reference link to the prompt
        prompt += f"{doc['content']} [Reference: {doc['link']}]\n"

    # Tokenize the input prompt with padding and truncation
    inputs = tokenizer(
        prompt,                       # The input prompt text
        return_tensors="pt",          # Return the tensors in PyTorch format
        padding=True,                 # Pad the input to the maximum length
        truncation=True,              # Truncate the input to fit within the maximum length
        max_length=500                # Set the maximum length for the tokenized input
    )

    # Move the input tensors to the GPU if available
    input_ids = inputs.input_ids.to(device) 

    # Move attention_mask to GPU if available 
    attention_mask = inputs.attention_mask.to(device)  

    # Generate the answer using the model
    output = gpt_model.generate(
        input_ids,                    # Input token IDs for the model
        attention_mask=attention_mask, # Attention mask for the model
        num_return_sequences=1,       # Generate only one answer sequence
        no_repeat_ngram_size=2,       # Prevent repeating n-grams of size 2
        max_new_tokens=100,           # Set the maximum number of new tokens to generate
        temperature=0.7,              # Set the temperature for sampling, controlling creativity
        top_k=10                      # Use top-k sampling to limit the number of token choices
    )

    # Decode the generated output tokens to a readable string
    answer = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    logging.info("Answer generated")

    # Return the generated answer
    return answer


def evaluate_rouge(generated_answer, reference_answer):
    """
    Evaluate the generated answer against a reference answer using ROUGE scores.
    """
     # Initialize a RougeScorer object to compute ROUGE scores (ROUGE-1 and ROUGE-L) with stemming
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # Compute the ROUGE scores between the reference answer and the generated answer
    scores = scorer.score(reference_answer, generated_answer)

    # Return the computed ROUGE scores
    return scores


# Retrieve cosine similarity score
def evaluate_cosine_similarity(generated_answer, reference_answer):
    """
    Evaluate the cosine similarity between the generated answer and a reference answer.
    """
    # Create a CountVectorizer to convert the text into a matrix of token counts
    vectorizer = CountVectorizer().fit_transform([generated_answer, reference_answer])

    # Convert the token count matrix into an array
    vectors = vectorizer.toarray()

    # Compute the cosine similarity between the vectors of the generated and reference answers
    cosine_sim = cosine_similarity(vectors)

    # Return the cosine similarity between the two documents (the generated and the reference answers)
    return cosine_sim[0, 1]  


def main_search_and_answer(index, chunks, query, k=5, reference_answer=""):
    """
    Perform a search in the FAISS index and generate an answer to the query.
    Evaluate the answer using ROUGE and cosine similarity metrics.
    """
    logging.info(f"Starting search and answer generation for query: {query}")

    # Treat the query itself as a chunk for generating embeddings
    query_chunks = [query]

    # Generate embeddings for the query
    query_embeddings = generate_embeddings(query_chunks)

    # Search the index for the top k most similar items to the query embeddings
    distances, indices = search_index(index, query_embeddings, k)

    # Retrieve the relevant documents based on the indices obtained from the search
    relevant_docs = []
    for idx in indices[0]:  # Iterate over the list of indices
        # Create a dictionary for each relevant document, including its content and a placeholder for the link
        relevant_doc = {
            'content': chunks[idx],  # Get the content of the document from the chunks based on the index
            'link': 'N/A'  # Placeholder for the document link (if available)
        }
        relevant_docs.append(relevant_doc) # Add the document to the list of relevant documents

    # Assume known_terms is a list of known terms extracted from your documents
    known_terms = set()  # Initialize an empty set for known terms (populate with actual terms as needed)

    # Generate an answer based on the query, relevant documents, and known term
    answer = generate_answer(query, relevant_docs, known_terms)

    # Log the completion of the search and answer generation process
    logging.info(f"Search and answer generation completed for query: {query}")
    
    # Evaluate the generated answer against the reference answer
    rouge_scores = evaluate_rouge(answer, reference_answer)  # Compute ROUGE scores
    cosine_sim = evaluate_cosine_similarity(answer, reference_answer)  # Compute cosine similarity

    # Return the generated answer along with evaluation score
    return answer, rouge_scores, cosine_sim


#####################
### Example usage ###
#####################

# Define the path to the folder containing the markdown files
folder_path = FILE_PATH

# Rebuild the FAISS index using the documents in the specified folder
# The 'rebuild_index' function processes the files and generates embeddings to create a new FAISS index
index, chunks = rebuild_index(folder_path)

# Define the query to be searched
query = "What is SageMaker?"

# Define the reference answer for evaluation
reference_answer = "Amazon SageMaker is a fully managed service that allows developers to build, train, and deploy machine learning models quickly. It provides tools for every step of the machine learning lifecycle, including data preparation, model training, tuning, and deployment. SageMaker also integrates with other AWS services and offers built-in algorithms, pre-built machine learning frameworks, and support for custom algorithms."

# Generate an answer based on the query and relevant documents from the FAISS index
# 'main_search_and_answer' function searches the index, generates an answer, and evaluates it against the reference answer
answer, rouge_scores, cosine_sim = main_search_and_answer(index, chunks, query, k=5, reference_answer=reference_answer)

# Print the generated answer, ROUGE scores and cosine similarity socre to the console
print(answer)
print("Rouge", rouge_scores)
print("Cosine", cosine_sim)