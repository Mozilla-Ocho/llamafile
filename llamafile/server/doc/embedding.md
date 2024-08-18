# LLaMAfiler Embedding Endpoint

The `/embedding` endpoint of our HTTP server provides a powerful way to
transform textual prompts into numerical representations. When you send
a POST request to this endpoint with a prompt in the request body, it
utilizes advanced natural language processing techniques to convert the
text into a high-dimensional vector. This vector, represented as an
array of floating-point numbers, captures the semantic essence of the
input prompt. Each float in the array corresponds to a dimension in the
embedding space, allowing for complex relationships between words and
concepts to be mathematically represented. These embeddings can be used
for various downstream tasks such as semantic search, text
classification, or content recommendation systems.

## Request URIs

- `/embedding` (llama.cpp compatible)
- `/v1/embeddings` (OpenAI compatible)

## Request Methods

- `POST` (recommended)
- `GET`

## Request Content Types

- `application/x-www-form-urlencoded` may be used to send parameters
  within the HTTP message body. This is useful for HTML forms.

- `text/plain` in which case the prompt parameter is supplied in the
  HTTP message body. It must be encoded using UTF-8. Please note the
  message body is only consulted if the prompt wasn't supplied as an
  HTTP GET or POST parameter.

- `application/json` in which case the HTTP message body must hold a
  JSON object, whose keys are the request parameters below. Please note
  the message body is only consulted if the prompt wasn't supplied as an
  HTTP GET or POST parameter.

## Request Parameters

- `content` (string) holds the prompt for which embeddings are
  calculated.
  
  This parameter may be passed as a GET parameter, e.g.
  `/embedding?content=orange`. It may be passed as a POST parameter. It
  may also be passed via a JSON object.
  
  This prompt is tokenized automatically. Each model has its own context
  window size that was used during training. The prompt is truncated if
  it has more tokens than the model supports. Truncation can be detected
  by checking if the `tokens_used` response parameter is less than
  `tokens_provided`. The `/tokenize` endpoint may also be used to check
  beforehand how the model chops up strings and into how many pieces.

- `input` (string) is an alias for `content`, which is provided for
  OpenAI API compatibility.

- `prompt` (string) is an alias for `content`, which is provided for
  consistency with the `/tokenize` endpoint.

- `add_special` (bool; default: true) may be specified to indicate if
  the tokenizer should insert special tokens automatically. What tokens
  get inserted, depends on the model architecture. For example,
  `all-MiniLM-L6-v2` likes to have a `[CLS]` at the beginning of the
  prompt, and a `[SEP]` token at the end. As such, these token are
  inserted implicitly by default.

- `parse_special` (bool; default: false) may be specified to indicate
  that the syntax for special tokens should be parsed in the prompt
  `content` parameter. For example, `all-MiniLM-L6-v2` defines the
  special token `[CLS]`. Under normal circumstances, this will be
  tokenized as literal text, i.e. `[" [", " cl", "s", " ]"]`, but if
  this parameter is true, then it'll be recognized as a single token.

## See Also

- [LLaMAfiler Documentation Index](index.md)
- [LLaMAfiler Endpoints Reference](endpoints.md)
- [LLaMAfiler Technical Details](technical_details.md)
