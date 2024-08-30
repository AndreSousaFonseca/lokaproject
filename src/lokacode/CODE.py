import os
import aiofiles
import asyncio
import nest_asyncio
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # Import torch for CUDA support
import re
import markdown
from bs4 import BeautifulSoup
import logging
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preprocessing function to clean and structure markdown content
def preprocess_markdown(content):
    logging.info("Preprocessing markdown content")
    
    def format_https_links(text):
        pattern = r'https://[^\s"\']+'
        return re.sub(pattern, lambda x: f'[Reference: {x.group()}]', text)
    
    content = format_https_links(content)
    content = re.sub(r'^(#{1,6})\s+', '', content, flags=re.MULTILINE)
    
    soup = BeautifulSoup(content, 'html.parser')
    preprocessed_text = soup.get_text()
    
    logging.info("Finished preprocessing markdown content")
    return preprocessed_text

# Step 1: Load and chunk the Markdown files asynchronously
async def process_md_file(file_path, chunk_size=512):
    logging.info(f"Processing file: {file_path}")
    
    async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
        content = await file.read()

    # Preprocess the content to clean and structure it
    content = preprocess_markdown(content)
    
    # Chunk the content
    tokens = content.split()
    chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    
    logging.info(f"File processed and chunked into {len(chunks)} chunks")
    return chunks

async def process_files(folder_path, chunk_size=512):
    logging.info(f"Processing files in folder: {folder_path}")
    
    all_chunks = []
    tasks = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            tasks.append(process_md_file(file_path, chunk_size))

    results = await asyncio.gather(*tasks)

    for chunks in results:
        all_chunks.extend(chunks)

    logging.info(f"All files processed. Total chunks: {len(all_chunks)}")
    return all_chunks

# Step 2: Perform Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    logging.info("Generating embeddings")
    embeddings = model.encode(chunks, show_progress_bar=True)
    logging.info("Embeddings generated")
    return embeddings

# Step 3: Create the FAISS index
def create_faiss_index(embeddings):
    logging.info("Creating FAISS index")
    
    embeddings_np = np.array(embeddings).astype(np.float32)
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    
    logging.info("FAISS index created")
    return index

def search_index(index, query_embeddings, k=5):
    logging.info(f"Searching index for top {k} results")
    distances, indices = index.search(query_embeddings, k)
    logging.info("Search completed")
    return distances, indices

# Rebuild the FAISS index
def rebuild_index(folder_path, chunk_size=512):
    logging.info("Rebuilding FAISS index")

    chunks = asyncio.run(process_files(folder_path, chunk_size))
    embeddings = generate_embeddings(chunks)

    dim = embeddings.shape[1]
    new_index = faiss.IndexFlatL2(dim)
    new_index.add(np.array(embeddings).astype(np.float32))

    logging.info("FAISS index rebuilt successfully")
    return new_index

# Step 4: Load the GPT model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gpt_model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt_model.to(device)

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
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def handle_oov_terms(question, known_terms):
    words = question.split()
    new_words = []
    for word in words:
        if word.lower() not in known_terms:
            synonyms = get_synonyms(word)
            if synonyms:
                new_words.append(synonyms[0])  # Use the first synonym as fallback
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def generate_answer(question, relevant_docs, known_terms):
    logging.info(f"Generating answer for question: {question}")

    # Handle OOV terms in the question
    question = handle_oov_terms(question, known_terms)
    
    prompt = few_shot_prompt.format(question=question)
    for doc in relevant_docs:
        prompt += f"{doc['content']} [Reference: {doc['link']}]\n"

    # Tokenize the input prompt with padding and truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=500
    )

    input_ids = inputs.input_ids.to(device)  # Move input_ids to GPU if available
    attention_mask = inputs.attention_mask.to(device)  # Move attention_mask to GPU if available

    # Generate the answer using the model
    output = gpt_model.generate(
        input_ids,
        attention_mask=attention_mask,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_new_tokens=100,
        temperature=0.7,
        top_k=10
    )

    # Decode and return the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    logging.info("Answer generated")
    return answer

def main_search_and_answer(index, query, k=5):
    logging.info(f"Starting search and answer generation for query: {query}")

    query_chunks = [query]  # Treat the query as a chunk for embedding
    query_embeddings = generate_embeddings(query_chunks)
    distances, indices = search_index(index, query_embeddings, k)

    # Retrieve the relevant documents
    relevant_docs = []
    for idx in indices[0]:  # Assuming indices is a list of indices
        relevant_doc = {
            'content': chunks[idx],  # Retrieve content from chunks based on indices
            'link': 'N/A'  # Placeholder if no link is available
        }
        relevant_docs.append(relevant_doc)

    # Assume known_terms is a list of known terms extracted from your documents
    known_terms = set()  # Populate this set with terms from your documents
    answer = generate_answer(query, relevant_docs, known_terms)

    logging.info(f"Search and answer generation completed for query: {query}")
    return answer

# Example usage
folder_path = r"C:\Users\andre\1.lokacode\sagemaker_documentation"
index = rebuild_index(folder_path)
query = "What is SageMaker?"
answer = main_search_and_answer(index, query)
print(answer)
