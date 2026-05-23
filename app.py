import os
import json
import asyncio
import litellm
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# Load variables from the hidden .env file into the system configuration context
load_dotenv()

# =====================================================================
# 🛠️ THE MONKEYPATCH (Fixes background CrewAI caching anomalies)
# =====================================================================
original_completion = litellm.completion
original_acompletion = litellm.acompletion

def strip_cache_breakpoint(kwargs):
    if 'messages' in kwargs:
        for message in kwargs['messages']:
            if isinstance(message, dict):
                message.pop('cache_breakpoint', None)

def patched_completion(*args, **kwargs):
    strip_cache_breakpoint(kwargs)
    return original_completion(*args, **kwargs)

async def patched_acompletion(*args, **kwargs):
    strip_cache_breakpoint(kwargs)
    return await original_acompletion(*args, **kwargs)

litellm.completion = patched_completion
litellm.acompletion = patched_acompletion

# =====================================================================
# 💾 LOCAL PERSISTENT MEMORY AUTOMATION LABELS
# =====================================================================
MEMORY_FILE = "jarvis_memory.json"

def load_long_term_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_long_term_memory(messages):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error backup operations: {str(e)}")

# =====================================================================
# 🔍 CUSTOM SURFING TOOL DEFINITION (With Valid Docstring)
# =====================================================================
@tool("brave_search")
def surf_the_web(query: str) -> str:
    """Surfs the web using search indexing engines to pull real-time data snippets regarding active job openings, tech stacks, or corporate information."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No real-time listing assets discovered for this query parameter configuration."
            
            summary = []
            for item in results:
                summary.append(f"Title: {item['title']}\nLink: {item['href']}\nSnippet: {item['body']}\n---")
            return "\n".join(summary)
    except Exception as e:
        return f"An infrastructure anomaly occurred while attempting to parse web metrics: {str(e)}"

# =====================================================================
# 🌐 STREAMLIT INTERFACE CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Jarvis Job Engine", page_icon="🤖", layout="wide")

st.title("🤖 Jarvis Career Core & Intelligence System")
st.caption("Active multi-agent terminal tracking Upwork, LinkedIn, and software market trends with Persistent Memory.")

# Sidebar Controls - Pull default configuration states from the system variables context
default_key = os.getenv("GROQ_API_KEY", "")
default_name = os.getenv("MASTER_NAME", "Enter your name")
default_title = os.getenv("MASTER_TITLE", "Enter your title")
default_skills = os.getenv("MASTER_SKILLS", "Enter your skills")
default_projects = os.getenv("MASTER_PROJECTS", "Enter your projects")

st.sidebar.header("⚙️ Core Processing Systems")
groq_api_key = st.sidebar.text_input(
    "Secure Connection Token (Groq API Key)", 
    value=default_key, 
    type="password"
)

st.sidebar.write("---")
st.sidebar.subheader("👤 Master Profile Context")
my_name = st.sidebar.text_input("Full Name", value=default_name)
my_title = st.sidebar.text_input("Professional Title", value=default_title)
my_skills = st.sidebar.text_area("Skill Core Matrix", value=default_skills)
my_projects = st.sidebar.text_area("Active Project Proofs", value=default_projects)

# Memory Reset Mechanism right in the UI
st.sidebar.write("---")
if st.sidebar.button("🧹 Clear Jarvis Memory Bank", type="secondary"):
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    st.session_state.messages = [
        {"role": "assistant", "content": "System memory wiped cleanly. New database cycle initialized, sir."}
    ]
    st.rerun()

# =====================================================================
# 💬 CHAT MEMORY CONTEXT LOOP
# =====================================================================
if "messages" not in st.session_state:
    long_term_records = load_long_term_memory()
    if long_term_records:
        st.session_state.messages = long_term_records
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "System online. Long-term memory logs successfully loaded. What are we tracking today, sir?"}
        ]

# Output existing conversations across the web app frame
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================================================
# 🧠 ASYNC CORE PROCESSING PIPELINE
# =====================================================================
async def run_jarvis_pipeline(user_prompt, conversation_history, skills, projects, api_key):
    os.environ["GROQ_API_KEY"] = api_key
    groq_llm = LLM(model="groq/llama-3.1-8b-instant", temperature=0.4)
    
    # Format recent history strings to pass as context constraints to the analyst
    history_snippet = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in conversation_history[-5:]])
    
    scout_agent = Agent(
        role="Intelligence Gathering Scout",
        goal="Surf web environments dynamically to aggregate hyper-targeted data assets and live marketplace metrics.",
        backstory=(
            "You are an automated web tracking interface engine. You take raw user requests, turn them into "
            "optimized web queries, search platform footprints, and cleanly extract contextual summaries."
        ),
        tools=[surf_the_web],
        llm=groq_llm,
        verbose=True
    )
    
    analyst_agent = Agent(
        role="Senior Executive Career Advisor",
        goal="Synthesize raw data points and cross-reference them directly against user profiles and long-term conversation history.",
        backstory=(
            "You are Jarvis, an advanced contextual assistant. You have full access to what you and the user discussed "
            "previously, allowing you to build on top of past answers seamlessly without repeating old information."
        ),
        llm=groq_llm,
        verbose=True
    )

    task_fetch = Task(
        description=(
            f"The user's direct input command is: '{user_prompt}'\n\n"
            "1. Analyze if this input actually requires a live web search (like tracking jobs, tech trends, or company data).\n"
            "2. If it is just a casual greeting, greeting acknowledgment, or small talk (like 'hi', 'hello', 'you up', 'ok thanks'), "
            "DO NOT execute the brave_search tool. Simply return the text: 'Casual chat detected; search bypassed.'\n"
            "3. If it DOES require a search, extract the clean keywords and execute the brave_search tool."
        ),
        expected_output="Clean real-time search engine data snippets or a conversational bypass note.",
        agent=scout_agent
    )

    task_synthesize = Task(
        description=(
            f"The user's direct prompt is: '{user_prompt}'\n\n"
            f"Here is the context of your past conversation turns for long-term reference:\n"
            f"==== START HISTORY ====\n{history_snippet}\n==== END HISTORY ====\n\n"
            f"Developer Profile Summary:\n- Stack Matrix: {skills}\n- Core Projects: {projects}\n\n"
            "STRICT ANSWERING GUIDELINES:\n"
            "- Talk directly to the user like an advanced conversational AI assistant (Jarvis).\n"
            "- Do NOT use static markdown headers like 'Strategic Brief' for every answer.\n"
            "- Address their request directly using your past interactions to provide continuity."
        ),
        expected_output="A fluid, personalized conversational response directly addressing the user's current request.",
        agent=analyst_agent
    )

    crew = Crew(
        agents=[scout_agent, analyst_agent],
        tasks=[task_fetch, task_synthesize],
        verbose=True,
        cache=False
    )
    
    return await crew.kickoff_async()

# =====================================================================
# ⌨️ LIVE USER INPUT PROCESSOR LOOP
# =====================================================================
if user_input := st.chat_input("Command Jarvis..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Consulting local database logs and initializing agent streams..."):
            try:
                # Pass full contextual state sequence into processing engines
                jarvis_response = asyncio.run(
                    run_jarvis_pipeline(user_input, st.session_state.messages[:-1], my_skills, my_projects, groq_api_key)
                )
                
                response_text = str(jarvis_response)
                st.write(response_text)
                
                # Update current state matrices and commit array directly to storage files
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                save_long_term_memory(st.session_state.messages)
                
            except Exception as e:
                st.error(f"An unexpected exception occurred: {str(e)}")