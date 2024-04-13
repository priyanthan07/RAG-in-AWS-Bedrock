import json
import os
import boto3
import sys

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## vector embedding and vector store
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa import RetrievalQA

## bedrock client
bedrock = boto3.client(service_name = "bedrock-runtime")

## embedding
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## data ingestion
def data_ingestion():
    Loader = PyPDFDirectoryLoader("data")
    Documents = Loader.load()

    ## defining the text splitter
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n","\n\n"], chunk_size = 10000, chunk_overlap = 1000)

    docs = text_splitter.split_documents(Documents)
    return docs

## create the vector store
def getVectorStore(docs):
    vector_store_faiss = FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vector_store_faiss.save_local("faiss_index")

## get llama2 model
def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={"temperature":0.8})
    return llm

## get titan model
def get_titan_llm():
    llm = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock, model_kwargs={"maxTokenCount":512})
    return llm

## get mistralAI model
def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-large-2402-v1:0", client=bedrock, model_kwargs={"max_tokens":512})
    return llm

prompt_template = """

Human: Use the following context to provide a concise answer to the question at the end but use atleast summarize with 
200 words with detailed explanations. If you don't know the answer,just say that you don't know, don't hallucinate.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response(llm, vector_store_faiss, query):
    response = RetrievalQA.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer=response({"query":query})
    return answer['result']
