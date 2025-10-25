from __future__ import annotations

import streamlit as st

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import PIL.Image

from utils import format_docs

# ------------------------------------------------
# initialize api and LLM model
# ------------------------------------------------

API_KEY = st.secrets["API_KEY"]

# define the model to use
MODEL = "accounts/yacchatbot/deployedModels/gemma3-27b-finetuned-iuh7mdom"
# MODEL = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"

@st.cache_resource
def load_llm():
    return init_chat_model(
        model=MODEL,
        model_provider="fireworks",
        api_key=API_KEY,
        model_kwargs={
            "max_tokens": 1024,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.6,
        },
    )

llm = load_llm()

# ------------------------------------------------
# load vectorstore for RAG
# ------------------------------------------------

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return FAISS.load_local(
        "streamlit/vector_stores/UvA_AUG_YAG_chatbot",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# ------------------------------------------------
# create profile received from URL parameters
# ------------------------------------------------

PROFILE_FIELDS = [
    # required
    {"key": "name", "label": "Name:"},
    {"key": "age", "label": "Age:"},
    {"key": "gender", "label": "Gender:"},
    {"key": "location_country", "label": "Country of residence:"},

    {"key": "location_city", "label": "City of residence:"},
    {"key": "study_subject", "label": "Study:"},
    {"key": "religion", "label": "Religion:"},
    {"key": "religion_importance", "label": "Importance of religion:"},
    {"key": "ethnicity", "label": "Ethnic/family origins:"},

    {"key": "informal_care_count", "label": "Number of adults (18+) cared for:"},
    {"key": "age_loved_one", "label": "Age(s) of loved one(s):"},
    {"key": "relation_loved_one", "label": "Relation to loved one(s):"},
    {"key": "condition_loved_one", "label": "Loved one(s) health condition(s):"},
    {"key": "duration_loved_one", "label": "Duration of loved one(s) condition:"},
    {"key": "care_time", "label": "Duration(s) of providing care:"},
    {"key": "most_care_time", "label": "Provides most care:"},
    {"key": "other_care", "label": "Number of paid care workers:"},
    {"key": "satisfaction", "label": "Satisfaction with relationship:"},

    {"key": "prefers_quick_response", "label": "Prefers quick response:"},
]

params = st.query_params

profile = {}

for field in PROFILE_FIELDS:
    key = field["key"]
    value = params.get(key, "N/A")

    # special case - prepend a space before name
    if key == "name":
        profile[key] = " " + value
    else:
        profile[key] = value

user_profile = " | ".join(
    f"{field['label']} {profile.get(field['key'], 'N/A')}"
    for field in PROFILE_FIELDS
)

# ------------------------------------------------
# miscellaneous loads
# ------------------------------------------------

@st.cache_resource
def load_pfps():
    return PIL.Image.open("streamlit/static/carebot_pfp.png"), \
        PIL.Image.open("streamlit/static/carebot_pfp_larger.png"), \
        PIL.Image.open("streamlit/static/user_pfp.png")

bot_pfp, bot_pfp_larger, user_pfp = load_pfps()

# ------------------------------------------------
# streamlit start
# ------------------------------------------------

st.set_page_config(page_title="Carebot", page_icon="❤️", layout="wide")

st.title("Carebot")
st.caption("I'm here to listen and help 🤗")

# init and show chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=m.get("avatar", "🤖")):
        st.markdown(m["content"])

# greet the user once at start
if "greeted" not in st.session_state:
    with st.chat_message("assistant", avatar=bot_pfp):
        st.markdown(f"Hi{profile['name']}, nice to meet you!")

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Hi{profile['name']}, nice to meet you!",
        "avatar": bot_pfp
    })

    st.session_state.greeted = True
    
# create chat input field 
if prompt := st.chat_input("What is on your mind?"):
    
    # store and display the current prompt.
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "avatar": user_pfp}
    )

    with st.chat_message("user", avatar=user_pfp):
        st.markdown(prompt)

    # convert session messages to langchain message objects
    session_messages = []

    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            session_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            session_messages.append(AIMessage(content=content))
        else:
            session_messages.append(HumanMessage(content=content))

    query = st.session_state.messages[-1]["content"]

    # retrieval process
    retrieved_docs = vectorstore.max_marginal_relevance_search(query, k=3, filter=None)

    # format retrieved docs
    context = format_docs(retrieved_docs)

    instruction = (
        "You are Carebot, a warm, empathic, supportive assistant. "
        "You speak kindly and clearly, avoid medical or legal claims, and focus on practical, everyday help. "
        "Reflect feelings, validate, suggest small steps and resources, and encourage seeking trusted adults or professionals when appropriate. "
        "Keep answers concise but caring. Use the user's profile to personalize your tone and suggestions:\n "
        f"{user_profile}\n\n"
        "If the user prefers a quick response, immediately give your advice. If not, employ scaffolding techniques and ask one or two questions"
        "to better understand the situation before giving advice. Keep this scaffolding process concise.\n\n"
        f"You may use the retrieved context to help you answer the question, if relevant:\n{context}\n\n"
        "Here is the user query:\n\n"
    )

    all_context = [SystemMessage(content=instruction)] + session_messages

    # generate
    with st.spinner("Thinking..."):
        response = llm.generate([all_context + [SystemMessage(content="\nKeep your response concise, except when asked for details\n")]])

    assistant_text = ""

    try:
        assistant_text = response.generations[0][0].text or ""
    except Exception:
        assistant_text = "Sorry, I couldn't generate a response. Try again later."

    # display
    with st.chat_message("assistant", avatar=bot_pfp):
        st.markdown(assistant_text)

    # append the message into history
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text, "avatar": bot_pfp}
    )
