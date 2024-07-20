# LLaMAfiler Tokenization Endpoint

The LLaMAfiler Tokenization Endpoint provides a robust interface for
converting text prompts into tokens, a crucial preprocessing step for
many natural language processing tasks. This endpoint supports flexible
input methods, including GET and POST requests, with various content
types to accommodate different integration needs. Users can customize
the tokenization process through parameters such as `add_special` and
`parse_special`, allowing for fine-tuned control over the output.
Whether you're building a chatbot, developing a text analysis tool, or
working on any NLP project, this endpoint offers a convenient and
powerful way to tokenize your text data.

## Request URIs

- `/tokenize`

## Request Methods

- `GET`
- `POST`

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

- `prompt` (string) holds the prompt which will be tokenized.

  This parameter may be passed as a GET parameter, e.g.
  `/tokenize?prompt=orange`. It may be passed as a POST parameter. It
  may also be passed via a JSON object.

- `add_special` (bool; default: true) may be specified to indicate if
  the tokenizer should insert special tokens automatically. What tokens
  get inserted, depends on the model architecture. For example,
  `all-MiniLM-L6-v2` likes to have a `[CLS]` at the beginning of the
  prompt, and a `[SEP]` token at the end. As such, these token are
  inserted implicitly by default.

- `parse_special` (bool; default: false) may be specified to indicate
  that the syntax for special tokens should be parsed in the `prompt`
  parameter. For example, `all-MiniLM-L6-v2` defines the special token
  `[CLS]`. Under normal circumstances, this will be tokenized as literal
  text, i.e. `[" [", " cl", "s", " ]"]`, but if this parameter is true,
  then it'll be recognized as a single token.

## See Also

- [LLaMAfiler Documentation Index](index.md)
- [LLaMAfiler Endpoints Reference](endpoints.md)
- [LLaMAfiler Technical Details](technical_details.md)
