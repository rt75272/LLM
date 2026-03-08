import streamlit as st
from rag_backend import get_rag_chain

# Initialize the Streamlit page
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖")
st.title("Enterprise RAG Assistant")
st.markdown("Ask questions about the uploaded internal documentation.")

# Initialize the RAG chain in session state so it doesn't reload on every keystroke
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_rag_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            response = st.session_state.rag_chain.invoke(prompt)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
