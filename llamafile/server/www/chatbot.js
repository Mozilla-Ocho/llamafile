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

const DEFAULT_SYSTEM_PROMPT =
      "A chat between a curious human and an artificial " +
      "intelligence assistant. The assistant gives helpful, " +
      "detailed, and polite answers to the human's questions.";

const DEFAULT_FLAGZ = {
  "model": null,
  "prompt": null,
  "no_display_prompt": false,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "temperature": 0.8,
  "top_p": 0.95,
  "seed": null
};

const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const stopButton = document.getElementById("stop-button");
const settingsButton = document.getElementById("settings-button");
const settingsModal = document.getElementById("settings-modal");
const closeSettings = document.getElementById("close-settings");

let abortController = null;
let disableAutoScroll = false;
let streamingMessageContent = [];
let uploadedFiles = [];
let chatHistory = [];
let flagz = null;

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

function createMessageElement(content, role) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", role);
  let hdom = new HighlightDom(messageDiv);
  const high = new RenderMarkdown(hdom);
  high.feed(content);
  high.flush();
  return messageDiv;
}

function scrollToBottom() {
  if (!disableAutoScroll)
    document.getElementById("bottom").scrollIntoView({behavior: "instant"});
}

function onChatInput() {
  chatInput.style.height = "auto";  // computes scrollHeight
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
}

function cleanupAfterMessage() {
  disableAutoScroll = false;
  chatMessages.scrollTop = chatMessages.scrollHeight;
  chatInput.disabled = false;
  sendButton.style.display = "inline-block";
  stopButton.style.display = "none";
  abortController = null;
  chatInput.focus();
}

function onWheel(e) {
  if (e.deltaY < 0)
    disableAutoScroll = true;
}

async function handleChatStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentMessageElement = createMessageElement("", "assistant");
  chatMessages.appendChild(currentMessageElement);
  let hdom = new HighlightDom(currentMessageElement);
  const high = new RenderMarkdown(hdom);
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
    if (error.name !== "AbortError") {
      console.error("Error reading stream:", error);
    }
  } finally {
    high.flush();
    cleanupAfterMessage();
  }
}

function stopMessage() {
  if (abortController) {
    abortController.abort();
    cleanupAfterMessage();
  }
}

function fixUploads(str) {
  str = uploadedFiles.reduce(
    (text, [from, to]) => text.replaceAll(from, to),
    str);
  uploadedFiles.length = 0;
  return str;
}

async function sendMessage() {
  const message = fixUploads(chatInput.value.trim());
  if (!message) return;

  // disable input while processing
  chatInput.value = "";
  chatInput.disabled = true;
  onChatInput();
  disableAutoScroll = false;
  sendButton.style.display = "none";
  stopButton.style.display = "inline-block";
  stopButton.focus();
  abortController = new AbortController();

  // add user message to chat
  const userMessageElement = createMessageElement(message, "user");
  chatMessages.appendChild(userMessageElement);
  scrollToBottom();

  // update chat history
  chatHistory.push({ role: "user", content: message });

  const settings = loadSettings();
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
        temperature: settings.temperature,
        top_p: settings.top_p,
        presence_penalty: settings.presence_penalty,
        frequency_penalty: settings.frequency_penalty,
        stream: true
      }),
      signal: abortController.signal
    });
    if (response.ok) {
      await handleChatStream(response);
      const lastMessage = streamingMessageContent.join("");
      chatHistory.push({ role: "assistant", content: lastMessage });
    } else {
      console.error("sendMessage() failed due to server error", response);
      chatMessages.appendChild(createMessageElement(
        `Server replied with error code ${response.status} ${response.statusText}`,
        "system"));
      cleanupAfterMessage();
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      console.error("sendMessage() failed due to unexpected exception", error);
      chatMessages.appendChild(createMessageElement(
        "There was an error processing your request.",
        "system"));
    }
    cleanupAfterMessage();
  }
}

function onDragBegin(e) {
  e.preventDefault();
  e.stopPropagation();
  chatInput.classList.add("drag-over");
}

function onDragEnd(e) {
  e.preventDefault();
  e.stopPropagation();
  chatInput.classList.remove("drag-over");
}

function onDrop(e) {
  const files = e.dataTransfer.files;
  [...files].forEach(onFile);
}

function onPaste(e) {
  const items = e.clipboardData.items;
  for (let item of items) {
    if (item.type.startsWith("image/")) {
      e.preventDefault();
      onFile(item.getAsFile());
      return;
    }
  }
}

// fixes image data uri
// - convert to jpg if it's not jpg/png/gif
// - reduce quality and/or downscale if too big
async function fixImageDataUri(dataUri, maxLength = 1024 * 1024) {
  const mimeMatch = dataUri.match(/^data:([^;,]+)/);
  if (!mimeMatch)
    throw new Error("bad image data uri");
  const mimeType = mimeMatch[1].toLowerCase();
  const supported = ["image/jpeg", "image/jpg", "image/png", "image/gif"];
  if (supported.includes(mimeType))
    if (dataUri.length <= maxLength)
      return dataUri;
  const lossless = ["image/png", "image/gif"];
  const quality = lossless.includes(mimeType) ? 0.92 : 0.8;
  function createScaledCanvas(img, scale) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(img.width * scale);
    canvas.height = Math.floor(img.height * scale);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "white";  // in case of transparency
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = async () => {
      let scale = 1.0;
      let attempts = 0;
      const maxAttempts = 5;
      const initialCanvas = createScaledCanvas(img, scale);
      let result = initialCanvas.toDataURL("image/jpeg", quality);
      while (result.length > maxLength && attempts < maxAttempts) {
        attempts++;
        scale *= 0.7071;
        const scaledCanvas = createScaledCanvas(img, scale);
        result = scaledCanvas.toDataURL("image/jpeg", quality);
        result.length = result.length;
      }
      if (result.length <= maxLength) {
        resolve(result);
      } else {
        reject(new Error(`Could not reduce image to ${(maxLength/1024).toFixed(2)}kb after ${maxAttempts} attempts`));
      }
    };
    img.onerror = () => {
      reject(new Error('Failed to load image from data URI'));
    };
    img.src = dataUri;
  });
}

async function onFile(file) {
  if (!file.type.toLowerCase().startsWith('image/')) {
    console.warn('Only image files are supported');
    return;
  }
  const reader = new FileReader();
  reader.onloadend = async function() {
    const description = file.name;
    const realDataUri = await fixImageDataUri(reader.result);
    const fakeDataUri = 'data:,placeholder/' + generateId();
    uploadedFiles.push([fakeDataUri, realDataUri]);
    insertText(chatInput, `![${description}](${fakeDataUri})`);
  };
  reader.readAsDataURL(file);
}

function insertText(elem, text) {
  const pos = elem.selectionStart;
  elem.value = elem.value.slice(0, pos) + text + elem.value.slice(pos);
  const newPos = pos + text.length;
  elem.setSelectionRange(newPos, newPos);
  elem.focus();
  elem.dispatchEvent(new Event("input"));
}

function onKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

async function fetchFlagz() {
  try {
    const response = await fetch("/flagz");
    return await response.json();
  } catch (error) {
    console.error("Could not fetch /flagz so using defaults", error);
    return DEFAULT_FLAGZ;
  }
}

function getSystemPrompt() {
  let defaultPrompt = flagz.prompt;
  if (!defaultPrompt)
    defaultPrompt = DEFAULT_SYSTEM_PROMPT;
  let promptsText = localStorage.getItem("v1.prompts");
  if (!promptsText)
    return defaultPrompt;
  let prompts = JSON.parse(promptsText);
  let prompt = prompts[defaultPrompt];
  if (!prompt)
    return defaultPrompt;
  return prompt;
}

function updateModelInfo() {
  if (flagz.model) {
    document.getElementById("model").textContent = flagz.model;
  }
}

function startChat(history) {
  chatHistory = history;
  chatMessages.innerHTML = "";
  for (let i = 0; i < chatHistory.length; i++) {
    if (flagz.no_display_prompt && chatHistory[i].role == "system")
      continue;
    chatMessages.appendChild(createMessageElement(chatHistory[i].content,
                                                  chatHistory[i].role));
  }
  scrollToBottom();
}

function loadSettings() {
  const stored = localStorage.getItem('v1.modelSettings');
  if (stored) {
    return JSON.parse(stored);
  }
  return {
    temperature: flagz.temperature,
    top_p: flagz.top_p,
    presence_penalty: flagz.presence_penalty,
    frequency_penalty: flagz.frequency_penalty,
  };
}

function saveSettings(settings) {
  localStorage.setItem('v1.modelSettings', JSON.stringify(settings));
}

function formatDoubleWithPlus(x) {
  return (x >= 0 ? "+" : "") + x.toFixed(2);
}

function updateSettingsDisplay(settings) {
  document.getElementById("temp-value").textContent = settings.temperature ? settings.temperature.toFixed(2) : "0.00 (deterministic)";
  document.getElementById("top-p-value").textContent = settings.top_p.toFixed(2);
  document.getElementById("presence-value").textContent = formatDoubleWithPlus(settings.presence_penalty);
  document.getElementById("frequency-value").textContent = formatDoubleWithPlus(settings.frequency_penalty);
  document.getElementById("temperature").value = settings.temperature;
  document.getElementById("top-p").value = settings.top_p;
  document.getElementById("presence-penalty").value = settings.presence_penalty;
  document.getElementById("frequency-penalty").value = settings.frequency_penalty;

  // Handle top-p disabling - using a more reliable selector
  const topPSettingItem = document.querySelector('.setting-item:has(#top-p)');
  if (settings.temperature === 0) {
    topPSettingItem.classList.add('disabled');
  } else {
    topPSettingItem.classList.remove('disabled');
  }

  // Update top-p description with percentage
  const topPDescription = topPSettingItem.querySelector('.setting-description');
  if (settings.top_p >= 1) {
    topPDescription.textContent = "Disabled. All tokens will be considered by the sampler.";
  } else if (settings.top_p > .5) {
    const percentage = Math.round((1 - settings.top_p) * 100);
    topPDescription.textContent = `The bottom ${percentage}% tokens will be ignored by the sampler.`;
  } else {
    const percentage = Math.round(settings.top_p * 100);
    topPDescription.textContent = `Only the top ${percentage}% tokens will be considered by the sampler.`;
  }
}

function setupSettings() {    
  settingsButton.addEventListener("click", () => {
    settingsModal.style.display = "flex";
    updateSettingsDisplay(loadSettings());
  });
  closeSettings.addEventListener("click", () => {
    settingsModal.style.display = "none";
  });
  ["temperature", "top-p", "presence-penalty", "frequency-penalty"].forEach(id => {
    const element = document.getElementById(id);
    element.addEventListener("input", (e) => {
      const settings = loadSettings();
      const value = parseFloat(e.target.value);
      const key = id.replace(/-/g, '_');
      settings[key] = value;
      saveSettings(settings);
      updateSettingsDisplay(settings);
    });
  });
  settingsModal.addEventListener("mousedown", (e) => {
    if (e.target === settingsModal) {
      settingsModal.style.display = "none";
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      settingsModal.style.display = "none";
    }
  });
}

async function chatbot() {
  flagz = await fetchFlagz();
  updateModelInfo();
  setupSettings();
  startChat([{ role: "system", content: getSystemPrompt() }]);
  sendButton.addEventListener("click", sendMessage);
  stopButton.addEventListener("click", stopMessage);
  chatInput.addEventListener("input", onChatInput);
  chatInput.addEventListener("keydown", onKeyDown);
  document.addEventListener("wheel", onWheel);
  document.addEventListener("dragenter", onDragBegin);
  document.addEventListener("dragover", onDragBegin);
  document.addEventListener("dragleave", onDragEnd);
  document.addEventListener("drop", onDragEnd);
  document.addEventListener("drop", onDrop);
  document.addEventListener("paste", onPaste);
  chatInput.focus();
}

chatbot();
