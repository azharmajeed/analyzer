import streamlit as st
import openai
import os
from dotenv import load_dotenv
from io import StringIO

load_dotenv()


st.title("🦜🔗 Quickstart App")

openai_api_key = os.environ.get("OPENAI_API_KEY")
uploaded_file = st.file_uploader("Choose a file")
print(uploaded_file)
print(type(uploaded_file))
print(st.session_state, "ss")

if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)
    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)
    print(string_data, "---string---")

# memory of previous converstion
# embeddings from uploaded documents
# ability to query a database
# ability to search the web
# custom prompt from the collation of the above results


# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         ):
#             full_response += response.choices[0].delta.get("content", "")
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
