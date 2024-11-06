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

function copyTextToClipboard(text) {
  // see https://stackoverflow.com/a/30810322/1653720
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(function() {
    }, function(err) {
      console.error('could not copy text: ', err);
    });
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
