#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab03

import streamlit as st
from openai import OpenAI

##***
# Part B additions:
#   Step 3  - conversation buffer: keep only the last 2 messages from the user
#             (and the LLM's responses to those messages)
#   Step 4a - calculate the total number of tokens passed to the LLM per request
#   Step 4b - token-based buffer: pass at most MAX_TOKENS to the LLM
#
# requirements.txt now also needs:
#     tiktoken
# tiktoken downloads its encoding file the first time it is used, so the
# deployed app needs outbound network access on that first token count.
# If tiktoken is unavailable the app still runs and falls back to an estimate.
try:
    import tiktoken
except ImportError:
    tiktoken = None
##***

model_to_use = "gpt-5.4-mini"

##***
# --- Part B settings: defined in the application, not exposed to the user ---
BUFFER_USER_TURNS = 2    # step 3: how many of the user's most recent messages to keep
MAX_TOKENS = 1000        # step 4b: ceiling on the tokens sent in one request
##***

st.title(":blue[Lab 3:] :grey[Deep] Chatbot")
st.write("Ask me anything!")

##***
# Both buffers from Part B are implemented below, so this radio decides which
# one is applied to the next request. This exists to make each part of the lab
# demonstrable side by side -- to hardwire one instead, delete the radio and
# set the variable directly, e.g.  buffer_mode = "Token limit"
buffer_mode = st.sidebar.radio(
    "Conversation buffer",
    ("Last 2 user turns", "Token limit", "No buffer (Part A)"),
    index=0,
)
st.sidebar.caption(
    f"Turns kept: {BUFFER_USER_TURNS}  |  Token ceiling: {MAX_TOKENS}"
)
##***

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

##***
# --- Token counting (step 4a) ----------------------------------------------

@st.cache_resource
def get_encoding():
    """Load the tokenizer once and reuse it across reruns."""
    if tiktoken is None:
        return None
    try:
        # tiktoken may not recognise a very new model name.
        return tiktoken.encoding_for_model(model_to_use)
    except Exception:
        # o200k_base is the encoding used by the recent GPT model families.
        # If this model's exact encoding differs, counts are close but not exact.
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
        # Fallback when tiktoken is not installed: roughly 4 characters per token.
        return len(text) // 4 + 4

    return len(encoding.encode(text)) + 4


def count_request_tokens(messages):
    """Total tokens for an entire request. The +3 is the priming the API adds
    when it asks the model for a reply."""
    return sum(count_message_tokens(m) for m in messages) + 3


# --- Buffers ----------------------------------------------------------------

def buffer_by_user_turns(messages, turns=BUFFER_USER_TURNS):
    """Step 3: keep only the last `turns` messages from the user, plus
    everything that came after the first of them -- which is exactly the LLM's
    responses to those messages.

    Slicing from the Nth-from-last user message means the buffer always starts
    with a user message and the user/assistant pairs stay intact.
    """
    user_indexes = [i for i, m in enumerate(messages) if m["role"] == "user"]

    # Not enough user messages yet, so there is nothing to trim.
    if len(user_indexes) <= turns:
        return list(messages)

    start = user_indexes[-turns]
    return messages[start:]


def buffer_by_tokens(messages, max_tokens=MAX_TOKENS):
    """Step 4b: walk backwards from the newest message, keeping messages while
    the running total stays within max_tokens.

    Newest messages are kept first because recent context matters most. The
    `if kept` guard means the newest message is always sent even when it alone
    exceeds the ceiling, since sending an empty list would fail the API call.
    """
    kept = []
    total = 3  # the reply priming that count_request_tokens also adds

    for msg in reversed(messages):
        msg_tokens = count_message_tokens(msg)

        if kept and total + msg_tokens > max_tokens:
            break

        kept.insert(0, msg)
        total += msg_tokens

    return kept
##***

##***
# The display loop below deliberately still walks the FULL history. The buffer
# controls what is sent to the model, not what the user sees, so the visible
# conversation is never truncated.
##***
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client

    ##***
    # Build the trimmed list that is actually sent to the LLM.
    if buffer_mode == "Last 2 user turns":
        messages_to_send = buffer_by_user_turns(st.session_state.messages)
    elif buffer_mode == "Token limit":
        messages_to_send = buffer_by_tokens(st.session_state.messages)
    else:
        # Part A behaviour: the whole conversation goes every time.
        messages_to_send = st.session_state.messages

    # Step 4a: record the token accounting so it can be shown in the sidebar.
    st.session_state.last_request_messages = len(messages_to_send)
    st.session_state.last_request_total = len(st.session_state.messages)
    st.session_state.last_request_tokens = count_request_tokens(messages_to_send)
    st.session_state.last_request_full_tokens = count_request_tokens(
        st.session_state.messages
    )
    ##***

    try:
        stream = client.chat.completions.create(
            model = model_to_use,
            #messages=st.session_state.messages,
            ##***
            messages=messages_to_send,
            ##***
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

    except Exception as e:
        st.error(f"This request has failed: {e}")
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})

##***
# Step 4a display: what the last request actually cost. Reading from
# session_state means the numbers survive reruns instead of vanishing.
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
##***
