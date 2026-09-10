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

st.title(":blue[Lab 3:] :grey[Deep] Chatbot")
st.write("Ask me anything!")

buffer_mode = st.sidebar.radio("Conversation buffer",
    ("Last 2 user turns", "Token limit", "No buffer (Part A)"),
    index=0,
)
st.sidebar.caption(
    f"Turns kept: {buffer}  |  Token ceiling: {max_tokens}"
)

if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]

    #try:
    #    api_key = st.secrets["OPENAI_API_KEY"]
    #except Exception:
    #    api_key = None

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


def buffer_by_tokens(messages, max_tokens=max_tokens):
    """Step 4b: walk backwards from the newest message, keeping messages while
    the running total stays within max_tokens.

    Newest messages are kept first because recent context matters most. The
    `if kept` guard means the newest message is always sent even when it alone
    exceeds the ceiling, since sending an empty list would fail the API call.
    """
    kept = []
    total = 3

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

    if buffer_mode == "Last 2 user turns":
        messages_to_send = buffer_by_user_turns(st.session_state.messages)
    elif buffer_mode == "Token limit":
        messages_to_send = buffer_by_tokens(st.session_state.messages)
    else:
        messages_to_send = st.session_state.messages

    st.session_state.last_request_messages = len(messages_to_send)
    st.session_state.last_request_total = len(st.session_state.messages)
    st.session_state.last_request_tokens = count_request_tokens(messages_to_send)
    st.session_state.last_request_full_tokens = count_request_tokens(
        st.session_state.messages
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
    st.sidebar.write(
        f"Messages: {st.session_state.last_request_messages} "
        f"of {st.session_state.last_request_total}"
    )
    st.sidebar.write(
        f"Tokens: {st.session_state.last_request_tokens} "
        f"(full history would be {st.session_state.last_request_full_tokens})"
    )
    if tiktoken is None:
        st.sidebar.caption("Estimated: tiktoken is not installed.")

#Indicate token based or user based buffer