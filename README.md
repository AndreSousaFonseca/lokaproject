LOKA_assessment

Here you will find my solution to the LOKA assessment code which revolved around the development of a Retrieval-Augmented Generation (RAG) model to retrieve information from a collection of markdown files and provide an answers to questions regaring the content of such files.

The code is structured into 4 main steps: Step 1 - Load and chunck the data into chunks of 512 tokens while preprocessing the markdown files Step 2 - Embedded the data within the chunks using the 'all-MiniLM-L6-v2' model Step 3 - Create a dense vector using FAISS (Facebook AI Similarity Search) Step 4 - Use the 'EleutherAI/gpt-neo-1.3B' model to produce the answer to a given question and evaluate the answer provided (rouge_score and cosine similarity score) Finally, I provide an example usage. 

The POC code can be found under the name CODE_POC.py, link: https://github.com/AndreSousaFonseca/lokaproject/blob/master/CODE_POC.py

I then extended this code to incorporate the AWS integration, CODE_AWS.py, https://github.com/AndreSousaFonseca/lokaproject/blob/master/CODE_AWS.py