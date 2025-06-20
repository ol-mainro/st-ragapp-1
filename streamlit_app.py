import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun


from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client


from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import warnings
from langchain._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_PRIVATE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")


embeddings = OpenAIEmbeddings()

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents_py",
    query_name="match_documents",
)

# query = "que veut dire TSE ?"
# matched_docs = vector_store.similarity_search(query)
# print(matched_docs)


st.title("🔎 L'humanologue")

"""
Je suis l'Humanologue, je sais tout sur les humains.

Ici nous pouvons évaluer une utilisation basique de l'approche RAG (Retrieval augmented Generation), 
pour améliorer les réponses d'un modèle de langue (LLM = Large Language Model), tel GPT d'OpenAI. 
La question de l'utilisateur est d'abord prise pour interroger une base de connaissances spécifique, 
puis le LLM est interrogé en lui demandant de prendre la réponse de la base spécifique en compte.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Je suis l'humanologue, comment vous aider ?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Qui est Charles Darwin ? Qui est Jean-François Dortier ? Qu'est ce qu'est la TSE ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=memory
        )

        result = conversation_chain.invoke({"question": prompt})
        answer = result["answer"]
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        messages = [
            (
            "system",
            prompt
            ),
        ]
        ai_msg = llm.invoke(messages)
        print(ai_msg.content)

        st.write('**Réponse avec la base de connaissance sur mesure (RAG):**')
        st.write(answer)
        st.write('**Réponse sans (GPT générique):**')
        st.write(ai_msg.content)
