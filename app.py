import streamlit as st
import rag_functions
from langchain.vectorstores.faiss import FAISS

st.title(" 🍁 RAG in AWS-Bedrock 🍁")

user_response = st.text_input("Question :")


with st.sidebar:
    st.title(" Vector Store:")

    if st.button("Vectors Update"):
        with st.spinner("Processing..."):
            docs = rag_functions.data_ingestion()
            rag_functions.getVectorStore(docs)
            st.success("Done")

if st.button("llama2 Output"):
    with st.spinner("Processing..."):
        faiss_index = FAISS.load_local("faiss_index", rag_functions.bedrock_embedding, allow_dangerous_deserialization=True)
        llm=rag_functions.get_llama2_llm()
        
        st.write(rag_functions.get_response(llm,faiss_index,user_response))
        st.success("Done")

if st.button("Titan Output"):
    with st.spinner("Processing..."):
        faiss_index = FAISS.load_local("faiss_index", rag_functions.bedrock_embedding, allow_dangerous_deserialization=True)
        llm=rag_functions.get_llama2_llm()
        
        st.write(rag_functions.get_response(llm,faiss_index,user_response))
        st.success("Done")

if st.button("mistral Output"):
    with st.spinner("Processing..."):
        faiss_index = FAISS.load_local("faiss_index", rag_functions.bedrock_embedding, allow_dangerous_deserialization=True)
        llm=rag_functions.get_mistral_llm()
        
        st.write(rag_functions.get_response(llm,faiss_index,user_response))
        st.success("Done")
