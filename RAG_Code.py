import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

model=OllamaLLM(model='llama3.1')

loader = TextLoader("C:/Users/prana/OneDrive/sample.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=5
)
texts = text_splitter.create_documents(docs[0].page_content)

db = FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.1"))
retriever = db.as_retriever()

prompt=ChatPromptTemplate.from_template(
"""
Based on the {context} provided answer the query asked by the user in a best possible way.
Example1- Question:"What skill is necessary to become Data Scientist?"
Answer:"SQL, Python, Machine Learning and concepts which help in future values predictions."
Question:{input}
Answer:
"""
)

combine_docs_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


result=retrieval_chain.invoke({'input':"In which industry do i work?"})
print(result["answer"])
