#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab03

import streamlit as st
from openai import OpenAI

try:
    import tiktoken
except ImportError:
    tiktoken = None

model_to_use = "gpt-5.4-mini"

buffer = 2
max_tokens = 500

SYSTEM_PROMPT = """You are a friendly explainer bot.

HOW TO WRITE:
- Explain everything so that a 10-year-old can understand it.
- Use short sentences and everyday words. Compare things to stuff a kid already knows.
- If you have to use a difficult word, explain what it means right away.
- Keep each answer to about 3 to 5 sentences.
- Do not use bullet points, headings, or emoji.

WHAT TO DO ON EACH TURN:
1. If the user asks a question, answer it, then end your message with exactly this line:
Do you want more info?
2. If the user says yes (or anything that means yes), give MORE detail about the same
topic you were just explaining. Add a new fact, an example, or a comparison you have
not used yet. Then end your message again with exactly this line:
Do you want more info?
3. If the user says no (or anything that means no), do not give more detail and do not
ask "Do you want more info?" again. Say something short and friendly, then ask what
else you can help with.
4. If the user sends a brand new question instead of yes or no, treat it as a new
question and follow rule 1.
5. If the user says yes but you cannot tell what the earlier topic was, ask them to
remind you what they want to hear more about. Never guess at the topic.
"""

st.title(":blue[Lab 3:] :grey[Deep] Chatbot")
st.write("Ask me anything!")

buffer_mode = st.sidebar.radio("Conversation buffer", ("Last 2 user turns", "Token limit"), index=0,)

st.sidebar.caption(f"Turns kept: {buffer}  |  Token ceiling: {max_tokens}")

if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]

    if not api_key:
        st.error("OPENAI_API_KEY invalid!")
        st.stop()

    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

@st.cache_resource
def get_encoding():
    """Load the tokenizer once and reuse it across reruns."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_to_use)
    except Exception:
        return tiktoken.get_encoding("o200k_base")

def count_message_tokens(msg):
    """Approximate the token cost of one message.
    The +4 accounts for the formatting the API wraps around every message
    (role markers and separators). It is an approximation, not an exact
    reproduction of OpenAI's internal accounting.
    """
    encoding = get_encoding()
    text = str(msg.get("role", "")) + str(msg.get("content", ""))

    if encoding is None:
        return len(text) // 4 + 4

    return len(encoding.encode(text)) + 4

def count_request_tokens(messages):
    """Total tokens for an entire request. The +3 is the priming the API adds
    when it asks the model for a reply."""
    return sum(count_message_tokens(m) for m in messages) + 3

def buffer_by_user_turns(messages, turns=buffer):
    """Step 3: keep only the last `turns` messages from the user, plus
    everything that came after the first of them -- which is exactly the LLM's
    responses to those messages.
    Slicing from the Nth-from-last user message means the buffer always starts
    with a user message and the user/assistant pairs stay intact.
    """
    user_indexes = [i for i, m in enumerate(messages) if m["role"] == "user"]

    if len(user_indexes) <= turns:
        return list(messages)

    start = user_indexes[-turns]
    return messages[start:]


def buffer_by_tokens(messages, max_tokens=max_tokens, reserved=0):
    """Step 4b: walk backwards from the newest message, keeping messages while
    the running total stays within max_tokens.

    Newest messages are kept first because recent context matters most. The
    `if kept` guard means the newest message is always sent even when it alone
    exceeds the ceiling, since sending an empty list would fail the API call.
    """
    kept = []

    total = 3 + reserved

    for msg in reversed(messages):
        msg_tokens = count_message_tokens(msg)

        if kept and total + msg_tokens > max_tokens:
            break

        kept.insert(0, msg)
        total += msg_tokens
    return kept

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    system_tokens = count_message_tokens(system_msg)

    if buffer_mode == "Last 2 user turns":
        messages_to_send = buffer_by_user_turns(st.session_state.messages)
    else:
        messages_to_send = buffer_by_tokens(
            st.session_state.messages, reserved=system_tokens
        )

    messages_to_send = [system_msg] + messages_to_send

    st.session_state.last_request_messages = len(messages_to_send)
    st.session_state.last_request_total = len(st.session_state.messages) + 1
    st.session_state.last_request_system = system_tokens
    st.session_state.last_request_mode = buffer_mode
    st.session_state.last_request_tokens = count_request_tokens(messages_to_send)

    st.session_state.last_request_full_tokens = count_request_tokens(
        [system_msg] + st.session_state.messages
    )

    try:
        stream = client.chat.completions.create(
            model = model_to_use,
            messages=messages_to_send,
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

    except Exception as e:
        st.error(f"This request has failed: {e}")
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})

if "last_request_tokens" in st.session_state:
    st.sidebar.divider()
    st.sidebar.write("**Last request sent to the LLM**")
    st.sidebar.write(f"Buffer used: {st.session_state.last_request_mode}")
    st.sidebar.write(
        f"Messages: {st.session_state.last_request_messages} "
        f"of {st.session_state.last_request_total}"
    )
    st.sidebar.write(
        f"Tokens: {st.session_state.last_request_tokens} "
        f"(full history would be {st.session_state.last_request_full_tokens})"
    )
    st.sidebar.caption(
        f"Of those, {st.session_state.last_request_system} tokens are the "
        f"system prompt, which is never trimmed."
    )
    if tiktoken is None:
        st.sidebar.caption("Estimated: tiktoken is not installed.")