import os  # System-dependent functionality like reading or writing
import json  # Import Json module
import boto3  #  AWS SDK for Python
import aiofiles # Asynchronous file operations
import asyncio  # Support for asynchronous programming
import re  # Support for regular expressions
from bs4 import BeautifulSoup  # Parsin HTML document
import logging  # Creates logging messages to help debbug
import numpy as np  # Handle large, multi-dimensional arrays and matrices
import faiss  # Similarity search and clustering of dense vectors
import torch  # Import torch for CUDA support
from sentence_transformers import SentenceTransformer  # Pre-trained models for transforming sentences into dense vector embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer  # Pre-trained models and tokenizers for natural language processing tasks
from nltk.corpus import wordnet  # Import WordNet lexical database for synonym Lookup
import nltk  # Tools for working with human language
from rouge_score import rouge_scorer  # Import ROUGE scorer
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity  # Import Cosine Similarity
from sagemaker import get_execution_role  # Retrieves the AWS IAM role that SageMaker uses to access AWS resources
from sagemaker.huggingface import HuggingFaceModel #Deploy Hugging Face models on SageMaker
from config import FILE_PATH, AWS_ID, ZIP_PATH, MODEL_PATH, PH_LINK  # Load the file, zip, model path and placeholder link  and AWS_ID from the config.py file

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Note loggings will not be commented

#AWS session using the boto3 library
session = boto3.Session(region_name='us-east-1')

# Client creation for AWS SageMaker service
sagemaker_client = session.client('sagemaker')

# Retireve the AWS IAM (Identity and Access Management) role 
role = get_execution_role()

# Preprocessing function
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
async def process_file(bucket_name, file_key, chunk_size=512):
    """
    Define an asynchronous function to process a file from an S3 bucket
    """
    logging.info(f"Processing file: {file_key}")
    
    # Retrieve the file from S3 using the provided bucket name and file key
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    
    # Read the content of the file and decode it from bytes to a UTF-8 string
    content = response['Body'].read().decode('utf-8')
   
    # Preprocess the Markdown content (e.g., convert Markdown to plain text)
    content = preprocess_markdown(content)
    
    # Split the preprocessed content into tokens (words)
    tokens = content.split()
    
    # Create chunks of the content, each with up to `chunk_size` tokens
    chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    
    logging.info(f"File processed and chunked into {len(chunks)} chunks")
    
    # Return the list of text chunks
    return chunks


lambda_client.create_function(
    """
    Lambda function deployment (Specify region in Lambda configuration
    """
    # Name of the Lambda function
    FunctionName='process_s3_files_and_transform',
    
    # Runtime environment for the Lambda function
    Runtime='python3.8',
    
    # IAM role ARN that the Lambda function will assume
    Role=f'arn:aws:iam::{AWS_ID}:role/my-lambda-role',
    
    # Handler function to call within your module
    Handler='lambda_function.lambda_handler',
    
    # Location of the deployment package in S3
    Code={
        'S3Bucket': bucket_name,
        'S3Key': ZIP_PATH,
    },
    
    # Environment variables for the Lambda function
    Environment={
        'Variables': {
            'AWS_REGION': 'us-east-1'  # Set the AWS region to ensure Lambda operates in the US region
        }
    }
)

def lambda_handler(event, context):
    """
    Define the Lambda handler function to process incoming S3 events
    """
    # Extract the bucket name from the S3 event record
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    
    # Extract the file key from the S3 event record
    file_key = event['Records'][0]['s3']['object']['key']
    
    # Get the current event loop for asynchronous tasks
    loop = asyncio.get_event_loop()
    
    # Process the file asynchronously and wait for completion
    chunks = loop.run_until_complete(process_file(bucket_name, file_key))
    
    # Return a success response indicating the file was processed successfully
    return {
        'statusCode': 200,
        'body': json.dumps('File processed successfully')
    }

# SageMaker Embedding Generation
# Retrieve the IAM role for SageMaker to use
role = get_execution_role()

# Create a HuggingFaceModel instance with the specified parameters
hf_model = HuggingFaceModel(
    
    # Location of the model data in S3
    model_data= MODEL_PATH,
    
    # IAM role for SageMaker
    role=role,
    
    # Version of the Transformers library to use
    transformers_version='4.11',
    
    # Version of PyTorch to use
    pytorch_version='1.9',
    
    # Python version for the environment
    py_version='py38'
)

# Deploy the Hugging Face model to a SageMaker endpoint
predictor = hf_model.deploy(
    # Type of instance to use for deployment
    instance_type='ml.g4dn.xlarge',
    
    # Name of the endpoint for the deployed model
    endpoint_name='embedding-endpoint',
    
    # Configuration name for the endpoint
    endpoint_config_name='your-endpoint-config',
    
    # AWS region where the endpoint will be deployed
    region_name='us-east-1'  # Ensure SageMaker is in the US region
)

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

# Define a function to secure an S3 bucket with encryption and access control
def secure_s3_bucket(bucket_name):
    # Enable encryption for new objects in the specified S3 bucket
    s3_client.put_bucket_encryption(
        Bucket=bucket_name,  # The name of the S3 bucket to apply encryption to
        ServerSideEncryptionConfiguration={  # Configuration for server-side encryption
            'Rules': [  # List of encryption rules
                {
                    'ApplyServerSideEncryptionByDefault': {  # Default encryption settings
                        'SSEAlgorithm': 'aws:kms',  # Use AWS Key Management Service (KMS) for encryption
                        'KMSMasterKeyID': 'your-kms-key-id'  # ID of the KMS key used for encryption
                    }
                }
            ]
        }
    )
    
    # Restrict bucket access to a specific IP range for added security
    s3_client.put_bucket_policy(
        Bucket=bucket_name,  # The name of the S3 bucket to apply the policy to
        Policy=json.dumps({  # Policy document in JSON format
            "Version": "2012-10-17",  # Version of the policy language
            "Statement": [  # List of policy statements
                {
                    "Effect": "Deny",  # Deny access to the bucket
                    "Principal": "*",  # Apply the policy to all users
                    "Action": "s3:*",  # Deny all actions on the bucket
                    "Resource": [  # Resources affected by the policy
                        f"arn:aws:s3:::{bucket_name}/*"  # ARN of all objects in the specified bucket
                    ],
                    "Condition": {  # Condition for applying the policy
                        "IpAddress": {"aws:SourceIp": "203.0.113.0/24"}  # Allow access only from this IP range (replace with your range)
                    }
                }
            ]
        })
    )

# Define a function to retrieve sensitive data from an S3 bucket
def get_sensitive_data(bucket_name, file_key):
    # Fetch the object from the specified S3 bucket
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    
    # Read the content of the object and decode it from bytes to a UTF-8 string
    content = response['Body'].read().decode('utf-8')
    
    # Return the content of the object
    return content


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
    input_ids = inputs.input_ids

    # Move attention_mask to GPU if available 
    attention_mask = inputs.attention_mask

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
    
    # Wrap the query into a list of chunks for embedding generation
    query_chunks = [query]
    
    # Generate embeddings for the query chunks
    query_embeddings = generate_embeddings(query_chunks)
    
    # Search the FAISS index for the top k results based on the query embeddings
    # `search_index` returns distances and indices of the nearest neighbors
    distances, indices = search_index(index, query_embeddings[0], k)
    
    # Initialize a list to store relevant documents
    relevant_docs = []

    # Loop over the indices of the nearest neighbors found in the FAISS index
    for idx in indices[0]:
        # Extract the content of the document corresponding to the index
        doc_content = chunks[idx]  # Assuming chunks contains the text of each document
        # Append the document content and a placeholder for the document link to the relevant_docs list
        relevant_docs.append({'content': doc_content, 'link': PH_LINK})

    # Generate an answer to the query based on the relevant documents and known terms
    answer = generate_answer(query, relevant_docs, known_terms=set())
    
    # If a reference answer is provided, evaluate the generated answer using ROUGE and cosine similarity metrics
    if reference_answer:
        # Calculate ROUGE scores to evaluate the quality of the generated answer
        rouge_scores = evaluate_rouge(answer, reference_answer)
        # Calculate cosine similarity to measure the similarity between the generated answer and the reference answer
        cosine_sim = evaluate_cosine_similarity(answer, reference_answer)
        # Log the ROUGE scores and cosine similarity results
        logging.info(f"ROUGE Scores: {rouge_scores}")
        logging.info(f"Cosine Similarity: {cosine_sim}")

    # Return a dictionary containing the generated answer and the list of relevant documents
    return {
        'answer': answer,
        'relevant_docs': relevant_docs
    }

if __name__ == "__main__":
    # Example usage
    # Define the path to the folder containing Markdown documentation files
    folder_path = r"C:\Users\andre\1.lokacode\sagemaker_documentation"

    # Rebuild the FAISS index using the documents in the specified folder
    # This function returns the index and the chunks of text from the documents
    index, chunks = rebuild_index(folder_path)

    # Define the query for which we want to generate an answer
    query = "How do I use AWS Lambda for asynchronous processing?"

    # Define the reference answer to evaluate the quality of the generated answer
    reference_answer = "AWS Lambda allows you to run code without provisioning or managing servers. It executes code only when triggered by specific events and automatically scales based on demand."

    # Perform the search and answer generation for the given query
    # The results include the generated answer and relevant documents
    results = main_search_and_answer(index, chunks, query, reference_answer=reference_answer)

    # Log the generated answer to the console
    logging.info(f"Generated Answer: {results['answer']}")

    # Log the relevant documents that were used to generate the answer
    logging.info(f"Relevant Documents: {results['relevant_docs']}")