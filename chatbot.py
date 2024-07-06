import pandas as pd
import os
import re
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("google_api_key")
genai.configure(api_key=os.getenv("google_api_key"))

# Directory containing the CSV files
input_folder = 'discord-chat'
output_file = 'output.txt'

# Initialize a list to store the formatted messages
formatted_messages = []

# Regular expression pattern to match the custom emoji format
emoji_pattern = re.compile(r'<:[a-zA-Z0-9_]+:[0-9]+>')

# Generate the output.txt file
def generate_output_file():
    # Loop through each CSV file in the directory
    for i in range(1, 18):
        # Construct the file name
        file_name = f'chat{i}.csv'
        file_path = os.path.join(input_folder, file_name)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Remove the 'Date' and 'User tag' columns
        df = df.drop(columns=['Date', 'User tag'])
        
        # Format each message and add it to the list
        for _, row in df.iterrows():
            content = row['Content']
            # Check if content is a string, otherwise set it to an empty string
            if not isinstance(content, str):
                content = ''
            
            # Remove custom emojis from the content
            content_cleaned = re.sub(emoji_pattern, '', content)
            
            if pd.notna(row['Mentions']):
                message = f"{row['Username']} said {content_cleaned} to {row['Mentions']}"
            else:
                message = f"{row['Username']} said {content_cleaned}"
            
            formatted_messages.append(message)
    
    # Write the formatted messages to the output file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        for message in formatted_messages:
            f.write(message + '\n')
    
    print(f"Formatted messages saved to {output_file}")

# Generate the output.txt file
generate_output_file()

# Functions for Streamlit app
def get_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    the context provided to you is a chat conversation where the first word of each line is the user who says the message and last word is the user who the message was meant to , these messages show you how the users typically are and their personality , give the correct response to the question asked by the user in the context , the context is : {context} , the question is : {question}
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat with Discord Logs")
    st.header("Chat with Discord Logs")

    user_question = st.text_input("Ask a Question from the Discord Logs")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        if st.button("Process Logs"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_file(output_file)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
