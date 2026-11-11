
import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

## LOAD FAISS VECTOR DATABASE
@st.cache_resource
def get_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db

#custom prompt template for the LLM

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context
to answer the user's question.

If you don't know the answer from the given context,
just say that you don't know. Do not make up information.

Do not provide anything outside the given context.



Context: {context}

Question: {question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(custom_prompt_template):

    prompt = PromptTemplate(template=custom_prompt_template,input_variables=["context", "question"])
    return prompt

#load llm

def load_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        max_tokens=512,
        api_key=GROQ_API_KEY
    )

    return llm



# MAIN STREAMLIT APPLICATION


def main():

    st.title("Medibot: Your Medical Chatbot")

    st.warning(
        "Educational information only. "
        "This chatbot does not replace professional medical advice, "
        "diagnosis, or treatment."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    prompt = st.chat_input("Ask your medical question here...")

    if prompt:

        # Display user question
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user question
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        try:

            with st.spinner("Searching medical documents and generating answer..."):

                # Load vector database
                vectorstore = get_vectorstore()

                if vectorstore is None:
                    st.error("Failed to load the vector store.")
                    st.stop()

                # Create RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": set_custom_prompt(
                            CUSTOM_PROMPT_TEMPLATE
                        )
                    }
                )

                # Generate response
                response = qa_chain.invoke({
                    "query": prompt
                })

                result = response["result"]

                source_documents = response["source_documents"]

            
            # DISPLAY ASSISTANT RESPONSE
            

            with st.chat_message("assistant"):

                st.markdown(result)

                # Display source documents
                with st.expander("Source Documents"):

                    for i, doc in enumerate(source_documents, 1):

                        st.write(f"### Source {i}")

                        st.write(
                            "File:",
                            doc.metadata.get("source", "Unknown")
                        )

                        st.write(
                            "Page:",
                            doc.metadata.get(
                                "page_label",
                                doc.metadata.get("page", "Unknown")
                            )
                        )

                        st.write("Retrieved Text:")

                        st.write(doc.page_content)

                        st.divider()

            # Save assistant response
            st.session_state.messages.append({"role": "assistant","content": result})

        except Exception as e:

            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
