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

const clipboardIcon = `<svg xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24" fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                      </svg>`;
const checkmarkIcon = `<svg xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24" fill="none"
                            stroke="green"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>`;

function createCopyButton(textProviderFunction, htmlProviderFunction) {
  if (!textProviderFunction) {
    throw new Error("textProviderFunction is null");
  }
  const copyButton = document.createElement('button');
  copyButton.className = 'copy-button';
  copyButton.innerHTML = clipboardIcon;
  copyButton.title = "Copy to clipboard";
  copyButton.addEventListener('click', async function () {
    try {
      await copyTextToClipboard(textProviderFunction(), htmlProviderFunction ? htmlProviderFunction() : null);
      copyButton.innerHTML = checkmarkIcon
      setTimeout(() => copyButton.innerHTML = clipboardIcon, 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  });
  return copyButton;
}

async function copyTextToClipboard(text, html) {
  // see https://stackoverflow.com/a/30810322/1653720
  if (navigator.clipboard) {
    if (html && window.ClipboardItem && navigator.clipboard.write) {
      const data = [
        new ClipboardItem({
          "text/html": new Blob([html], { type: "text/html" }),
          "text/plain": new Blob([text], { type: "text/plain" }),
        })
      ];
      return navigator.clipboard.write(data);
    } else {
      // Fallback to text-only copy if ClipboardItem is not supported (or html is not needed)
      return navigator.clipboard.writeText(text);
    }
  } else {
    var textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.top = '0';
    textArea.style.left = '0';
    textArea.style.position = 'fixed';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      if (!document.execCommand('copy')) {
        console.log('document.execCommand failed');
      }
    } catch (err) {
      console.error('document.execCommand raised: ', err);
    }
    document.body.removeChild(textArea);
  }
}
