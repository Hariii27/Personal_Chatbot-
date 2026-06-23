# =============================================================================
# PROJECT 1: Personal Q&A Chatbot with Memory
# Stack: Streamlit (UI) + LangGraph (Agent) + SqliteSaver (Memory)
# Deploy: Streamlit Community Cloud
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# =============================================================================
# SECTION 1: PAGE CONFIG
# This must be the FIRST Streamlit command in the file
# =============================================================================

st.set_page_config(
    page_title="Aria - Personal AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# =============================================================================
# SECTION 2: ENVIRONMENT SETUP
# In production (Streamlit Cloud), API keys are stored in Streamlit Secrets,
# NOT hardcoded in the file — this is the correct production pattern
# =============================================================================

# st.secrets reads from .streamlit/secrets.toml on local
# and from the Streamlit Cloud dashboard in production
os.environ["GROQ_API_KEY"] =st.secrets[ "GROQ_API_KEY"]

# =============================================================================
# SECTION 3: LangGraph AGENT SETUP
# st.cache_resource caches the agent across ALL user sessions
# This means the graph is compiled ONCE, not on every page reload
# Without this, the graph would be rebuilt on every single user interaction
# =============================================================================

@st.cache_resource
def build_agent():
    """
    Builds and compiles the LangGraph agent.
    Called only ONCE per app deployment due to @st.cache_resource.
    
    Returns the compiled LangGraph app ready to invoke.
    """
    
    # ── LLM Setup ────────────────────────────────────────────────────────────
    llm = Chatgroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        streaming=True    # Enables token-by-token streaming in the UI
    )
    
    # ── System Prompt ────────────────────────────────────────────────────────
    # This defines the chatbot's personality and behavior
    SYSTEM_PROMPT = SystemMessage(content="""
    You are Aria, a friendly and intelligent personal AI assistant.
    You have an excellent memory and always reference what the user has told you.
    If the user shares their name, preferences, or personal facts,
    remember them and use that information naturally in future responses.
    Keep your responses helpful, warm, and conversational.
    """)

    # ── Node Definition ──────────────────────────────────────────────────────
    # A NODE is a Python function that:
    #   INPUT  → current state (full message history)
    #   OUTPUT → dict with updated state fields
    def chatbot_node(state: MessagesState):
        # Prepend system prompt to the FULL conversation history
        # This is what gives the bot "memory" — it sees ALL past messages
        messages = [SYSTEM_PROMPT] + state["messages"]
        response = llm.invoke(messages)
        # add_messages reducer (built into MessagesState) APPENDS this response
        # to the existing messages list — never overwrites
        return {"messages": [response]}

    # ── Graph Construction ───────────────────────────────────────────────────
    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # ── Memory (Checkpointer) ────────────────────────────────────────────────
    # SqliteSaver stores conversation state in a local SQLite database file
    # This survives app restarts — production-appropriate for small-medium scale
    # For larger scale, replace with PostgresSaver
    import sqlite3
    conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    # ── Compile ──────────────────────────────────────────────────────────────
    # Attaching the checkpointer is what enables memory
    # Without checkpointer=memory, every invocation would be completely stateless
    app = graph.compile(checkpointer=memory)
    
    return app


# =============================================================================
# SECTION 4: STREAMLIT SESSION STATE
# Streamlit reruns the entire script on every user interaction.
# st.session_state persists data across these reruns for each user session.
#
# We use it to:
#   1. Store chat messages for display in the UI
#   2. Store a unique thread_id per user session
# =============================================================================

# Initialize chat history in session state if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []   # List of {"role": "user/assistant", "content": "..."}

# Generate a unique thread_id for this browser session
# This is the KEY that LangGraph uses to load/save the correct memory
# Each new browser tab/session gets its own isolated memory
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# Build the agent (cached — only happens once)
agent = build_agent()

# LangGraph config — thread_id tells the checkpointer WHOSE memory to load
config = {"configurable": {"thread_id": st.session_state.thread_id}}


# =============================================================================
# SECTION 5: UI LAYOUT
# =============================================================================

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🤖 Aria — Personal AI Assistant")
st.caption("I remember everything you tell me in this session.")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show current session ID (useful for debugging)
    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}...`")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        # Generate new thread_id to start a fresh memory in LangGraph
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    st.markdown("### 💡 Try asking:")
    st.markdown("- *My name is Thiru*")
    st.markdown("- *I love Python programming*")
    st.markdown("- *What's my name?*")
    st.markdown("- *What do I love?*")
    
    st.divider()
    st.markdown("### 🧠 How Memory Works")
    st.markdown("""
    Each message is saved to **SQLite** via LangGraph's checkpointer.
    Your `thread_id` is the key that loads your specific conversation history.
    """)


# =============================================================================
# SECTION 6: DISPLAY CHAT HISTORY
# Loop through stored messages and render them as chat bubbles
# =============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =============================================================================
# SECTION 7: HANDLE USER INPUT
# st.chat_input renders the input box at the bottom of the page
# It returns the user's message when they press Enter, or None otherwise
# =============================================================================

if user_input := st.chat_input("Message Aria..."):
    
    # ── Display user message immediately ─────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Invoke LangGraph Agent ────────────────────────────────────────────────
    with st.chat_message("assistant"):
        # st.spinner shows a loading indicator while the agent thinks
        with st.spinner("Aria is thinking..."):
            
            # Invoke the agent with just the NEW user message
            # LangGraph automatically loads the full history from SqliteSaver
            # using the thread_id in config
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
            
            # Extract the latest AI response (always the last message)
            ai_response = result["messages"][-1].content
        
        # Display the response
        st.markdown(ai_response)
    
    # ── Save assistant response to session state ──────────────────────────────
    # This keeps the UI in sync with LangGraph's internal state
    st.session_state.messages.append({"role": "assistant", "content": ai_response})


# =============================================================================
# CONCEPT RECAP — The Agentic Pattern in This App:
#
# USER INPUT
#     ↓
# Streamlit captures it → adds to session_state.messages (for UI display)
#     ↓
# agent.invoke() called with thread_id config
#     ↓
# LangGraph loads full history from SqliteSaver (using thread_id)
#     ↓
# START → chatbot_node (LLM sees system prompt + full history) → END
#     ↓
# LangGraph saves new state snapshot back to SqliteSaver
#     ↓
# AI response extracted → displayed in UI → saved to session_state
#
# KEY INSIGHT: LangGraph manages the "real" memory (in SQLite).
# st.session_state only stores messages for UI DISPLAY purposes.
# =============================================================================
