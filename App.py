css = '''
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex; /* Enable flexbox layout */
    }
   /* Add margins to align the boxes themselves */
    .chat-message.user {
        background-color: #2b313e;
        margin-right: auto; /* Pushes the box to the left */
    }

    .chat-message.bot {
        background-color: #475063;
        margin-left: auto; /* Pushes the box to the right */
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
</style>
'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''





import streamlit as st
from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from HtmlTemplate import css, bot_template, user_template
from InstructorEmbedding import INSTRUCTOR


def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        Pdf_reader = PdfReader(pdf)
        for page in Pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
            )
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl", model_kwargs ={"device": "cpu"})
    # Wrap text chunks in objects with 'page_content' attribute
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorstore):
    llms = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs ={"temperature": 0.3, "max_length":1000}, huggingfacehub_api_token="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")
    # llms = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = RetrievalQA.from_chain_type(
        chain_type="stuff",
        llm=llms,
        retriever= vectorstore.as_retriever(search_kwars={"k": 3}),
        memory = memory
    )

    return conversational_chain

def handle_userimput(user_question):
    response = st.session_state.conversation({'query': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, mesaage in enumerate (st.session_state.chat_history):
        if i % 2 == 0:
            st. write(user_template.replace("{{MSG}}", mesaage.content), unsafe_allow_html=True)
        else:
            st. write(bot_template.replace("{{MSG}}", mesaage.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Docs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Docs :books:")
    user_question = st.text_input("Ask question about your document")
    if user_question:
        handle_userimput(user_question)
    

    with st.sidebar:
        st.subheader("Your documents")
        docs =st.file_uploader("Upload your PDs here and click on 'process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                
                #get pdf text
                raw_text = get_pdf_text(docs)
                # st.write(raw_text)

                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()


