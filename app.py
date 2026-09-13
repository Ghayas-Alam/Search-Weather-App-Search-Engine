
import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


# ============================================================
# API KEY CHECK
# ============================================================

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing from your .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY is missing from your .env file.")
    st.stop()

if not WEATHER_API_KEY:
    st.error("❌ WEATHER_API_KEY is missing from your .env file.")
    st.stop()


# ============================================================
# TOOLS
# ============================================================

search_tool = TavilySearchResults(
    max_results=2
)


@tool
def get_weather_data(city: str) -> str:
    """Get current temperature, humidity, and air quality for a city."""

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_API_KEY}&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}."

        current = data["current"]

        temperature = current.get("temperature")
        humidity = current.get("humidity")
        air_quality = current.get("air_quality")

        return (
            f"Weather in {city}:\n"
            f"Temperature: {temperature}°C\n"
            f"Humidity: {humidity}%\n"
            f"Air Quality: {air_quality}"
        )

    except Exception as e:
        return f"Weather service error: {str(e)}"


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)


# ============================================================
# AGENT
# ============================================================

tools = [
    search_tool,
    get_weather_data
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# ============================================================
# SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.markdown("---")

    st.subheader("Available Tools")

    st.write("🌐 **Web Search**")
    st.caption("Search current information using Tavily.")

    st.write("🌤️ **Weather**")
    st.caption("Get temperature, humidity and air quality.")

    st.write("🧠 **Gemini AI**")
    st.caption("Generate intelligent answers.")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="title">🤖 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions, search the web, or check the weather.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_input:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke(user_input)

                answer = response["output"]

            except Exception as e:

                answer = (
                    "❌ Something went wrong.\n\n"
                    f"Error: `{str(e)}`"
                )

            st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

