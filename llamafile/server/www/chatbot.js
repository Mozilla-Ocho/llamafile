// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const API_ENDPOINT = "/v1/chat/completions";
const API_KEY = "your-api-key-here";

const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const typingIndicator = document.getElementById("typing-indicator");

let streamingMessageContent = [];

let chatHistory = [
  {
    role: "system",
    content: ("A chat between a curious human and an artificial " +
              "intelligence assistant. The assistant gives helpful, " +
              "detailed, and polite answers to the human's questions.")
  },
];

function createMessageElement(content, role) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", role);
  let hdom = new HighlightDom(messageDiv);
  const high = new HighlightMarkdown(hdom);
  high.feed(content);
  high.flush();
  return messageDiv;
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setTypingIndicatorVisibility(visible) {
  typingIndicator.style.display = visible ? "flex" : "none";
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function handleChatStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentMessageElement = createMessageElement("", "assistant");
  chatMessages.appendChild(currentMessageElement);
  let hdom = new HighlightDom(currentMessageElement);
  const high = new HighlightMarkdown(hdom);
  streamingMessageContent = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");

      // process all complete lines
      for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i].trim();
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") continue;

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content || "";
            streamingMessageContent.push(content);
            high.feed(content);
            scrollToBottom();
          } catch (e) {
            console.error("Error parsing JSON:", e);
          }
        }
      }

      // keep the last incomplete line in the buffer
      buffer = lines[lines.length - 1];
    }
  } catch (error) {
    console.error("Error reading stream:", error);
  } finally {
    high.flush();
    setTypingIndicatorVisibility(false);
    sendButton.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
  }
}

async function sendMessage() {
  const message = chatInput.value.trim();
  if (!message) return;

  // disable input while processing
  chatInput.value = "";
  chatInput.disabled = true;
  sendButton.disabled = true;

  // add user message to chat
  const userMessageElement = createMessageElement(message, "user");
  chatMessages.appendChild(userMessageElement);
  scrollToBottom();

  // update chat history
  chatHistory.push({ role: "user", content: message });

  // show typing indicator
  setTypingIndicatorVisibility(true);

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: chatHistory,
        stream: true
      })
    });

    if (!response.ok)
      throw new Error(`HTTP error! status: ${response.status}`);

    // handle the stream
    await handleChatStream(response);

    // update chat history with response
    const lastMessage = streamingMessageContent.join('');
    chatHistory.push({ role: "assistant", content: lastMessage });

  } catch (error) {
    console.error("Error:", error);
    const errorMessage = createMessageElement(
      "Sorry, there was an error processing your request.",
      "system");
    chatMessages.appendChild(errorMessage);
    setTypingIndicatorVisibility(false);
    sendButton.disabled = false;
    chatInput.disabled = false;
  }
}

// setup chat window
chatMessages.innerHTML = "";
for (let i = 0; i < chatHistory.length; i++) {
  chatMessages.appendChild(createMessageElement(chatHistory[i].content,
                                                chatHistory[i].role));
}

// setup events
sendButton.addEventListener("click", sendMessage);
chatInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// focus input
chatInput.focus();
