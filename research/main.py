
# Imports
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
# Environment
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


# ============================================================
# Check API keys
# ============================================================

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY is missing from .env")


# ============================================================
# Tavily Search Tool
# ============================================================

search_tool = TavilySearchResults(
    max_results=2
)


# ============================================================
# Weather Tool
# ============================================================

@tool
def get_weather_data(city: str) -> str:
    """Get current temperature, humidity, and air quality for a city."""

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

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


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)


# ============================================================
# Agent
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
# Test 1 - News
# ============================================================

response = agent.invoke(
    "Tell me the latest news about Iran and USA"
)

print(response["output"])


# ============================================================
# Test 2 - Weather
# ============================================================

response = agent.invoke(
    "What is the weather in Kolkata?"
)

print(response["output"])
