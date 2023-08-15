import os

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.indexes import VectorstoreIndexCreator

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
#import unstructured
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from langchain import PromptTemplate
from langchain import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain import ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
#import chromadb
#from chromadb.config import Settings
import json

#from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain.document_loaders.json_loader import JSONLoader

from langchain.document_loaders import DirectoryLoader, TextLoader
import streamlit as st



if 0:
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="content/"))

    collection = client.create_collection("my_information23")

    collection.add(
        documents=["This is a document containing car information",
        "This is a document containing information about dogs", 
        "This document contains four wheeler catalogue"],
        metadatas=[{"source": "Car Book"},{"source": "Dog Book"},{'source':'Vechile Info'}],
        ids=["150", "300", "10"]
    )

    results = collection.query(
        query_texts=["bus"],
        n_results=1
    )

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#doc =  Document(page_content=v['ans'], metadata={"source": "local"})
loader = JSONLoader(
    file_path='./stack-queue.json',
    jq_schema='.items[].ans')

data1 = loader.load()

loader = JSONLoader(
    file_path='./stack-queue.json',
    jq_schema='.items[].marks')

meta1 = loader.load()

db1 = FAISS.from_documents(data1, embeddings)

student_q1 = st.text_input ("What is the difference between a stack and a queue in programming?")

loader = JSONLoader(
    file_path='./quadratic.json',
    jq_schema='.items[].ans')

data2 = loader.load()

loader = JSONLoader(
    file_path='./quadratic.json',
    jq_schema='.items[].marks')

meta2 = loader.load()

db2 = FAISS.from_documents(data2, embeddings)

#student_q2 = st.text_input ("Solve the quadratic equation x^2 - 5x + 6 = 0")

#student_q1 = "A stack is a data structure that allows first-in, first-out while a queue is a datastructure that allows first-in, last-out"student_q1 = "A stack is a pyjama, a queue is a cot"
#student_q2 = "x=6 and x=5"
matching_docs = db1.similarity_search_with_score(student_q1, 1)
print ("Similarity Score =", matching_docs[0][1])

if (matching_docs[0][1] < 0.25):
    score = meta1[matching_docs[0][0].metadata['seq_num']-1].page_content
else:
    score = 0

print ("Recommended Marks =", score)

if (st.button("Evaluate")):
    st.write ("Your score is", score, "out of 5")
    if (score > 0):
        st.write ("Similarity Score is", matching_docs[0][1], "Index in training file is", matching_docs[0][0].metadata['seq_num'])

