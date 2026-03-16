import streamlit as st
import time
from dotenv import load_dotenv

# Import the actual agent functions from agents.py
from src.agents import run_single_agent, run_multi_agent

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Finance AI Agents", layout="wide")

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Agent Configuration")

agent_choice = st.sidebar.radio(
    "Select Agent Architecture:",
    ["Single Agent", "Multi-Agent"],
    help="Single Agent: all tools in one LLM. Multi-Agent: specialists + critic."
)

model_choice = st.sidebar.radio(
    "Select Model:",
    ["gpt-4o-mini", "gpt-4o"],
    help="gpt-4o-mini: faster, cheaper. gpt-4o: larger, more capable."
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Architecture Info:**
- **Single Agent:** One LLM with all 7 tools
- **Multi-Agent:** 3 specialists (market, fundamentals, sentiment) + critic

**Model Info:**
- **gpt-4o-mini:** $0.05K / ~17.5s per query
- **gpt-4o:** $0.15K / ~7.7s per query
""")

# ─────────────────────────────────────────────────────────────
# SESSION STATE: CONVERSATION HISTORY
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ─────────────────────────────────────────────────────────────
st.title("💰 Financial AI Agent Chat")
st.markdown(f"""
**Current Configuration:** `{agent_choice}` + `{model_choice}`

Ask financial questions about stocks, sectors, or market data.
The agent will use real-time tools to retrieve accurate data.
""")

# Display conversation history
st.markdown("---")
st.subheader("Conversation History")

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Agent:** {msg['content']}")
        if "metadata" in msg:
            meta = msg["metadata"]
            st.caption(f"🤖 {meta['architecture']} | {meta['model']} | ⏱️ {meta['time']}s | Tools: {meta['tools']}")
    st.markdown("")

# ─────────────────────────────────────────────────────────────
# USER INPUT & AGENT RESPONSE
# ─────────────────────────────────────────────────────────────
st.markdown("---")

with st.form(key=f"chat_form_{len(st.session_state.messages)}", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g., 'What is NVDA's P/E ratio?'",
            label_visibility="collapsed",
        )
    with col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)

if submit_button and user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call appropriate agent with conversation history as context
    with st.spinner("🤔 Thinking..."):
        try:
            t0 = time.time()

            # Build conversation context for multi-turn understanding
            conversation_context = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in st.session_state.messages[:-1]  # Exclude current user message (already in question)
            ])

            # Add context to question if there's prior history
            if conversation_context:
                full_question = f"Prior conversation:\n{conversation_context}\n\nCurrent question: {user_input}"
            else:
                full_question = user_input

            # Call actual agents with tools
            if agent_choice == "Single Agent":
                result = run_single_agent(full_question, model=model_choice, verbose=False)
                agent_response = result.answer
                tools_used = result.tools_called
                architecture = "Single Agent"
            else:  # Multi-Agent
                result = run_multi_agent(full_question, model=model_choice, verbose=False)
                agent_response = result["final_answer"]
                tools_used = [t for r in result["agent_results"] for t in r.tools_called]
                architecture = result["architecture"]

            elapsed = round(time.time() - t0, 2)
            tools_str = ", ".join(set(tools_used)) if tools_used else "none"

            # Add agent response to history with metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": agent_response,
                "metadata": {
                    "architecture": architecture,
                    "model": model_choice,
                    "time": elapsed,
                    "tools": tools_str
                }
            })

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.messages.pop()  # Remove failed user message

# ─────────────────────────────────────────────────────────────
# EXAMPLE FOLLOW-UP CHAINS (for testing)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("💡 Example Follow-up Chains to Test"):
    st.markdown("""
**Chain 1: Cross-Domain (Price → Fundamentals)**
1. "What is NVIDIA's 1-year stock performance?"
2. "How does that compare to AMD?"
3. "Which of the two has better P/E ratios?"

**Chain 2: Sector Analysis**
1. "List the top 3 semiconductor companies by 1-year return"
2. "What are their P/E ratios?"
3. "Which has the best sentiment in the news right now?"

**Chain 3: Multi-Condition**
1. "Which large-cap tech stocks have grown >20% this year?"
2. "Among those, which are on NASDAQ?"
3. "What's the news sentiment for the top performer?"
""")

st.markdown("---")
st.caption("Streamlit Finance Agent Chat | Built with OpenAI + Financial Data Tools")
