const xhr = new XMLHttpRequest();
xhr.open('GET', 'test.md');

xhr.onload = function() {
  if (xhr.status == 200) {
    let code = document.getElementById('code');
    let hdom = new HighlightDom(code);
    const h = new RenderMarkdown(hdom);
    h.feed(xhr.responseText);
    h.flush();
  } else {
    throw new Error(`HTTP error! status: ${xhr.status}`);
  }
};

xhr.onerror = function() {
  throw new Error('Network error');
};

xhr.send();
