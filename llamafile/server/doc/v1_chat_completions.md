# LLaMAfiler Chat Completions Endpoint

The `/v1/chat/completions` endpoint lets you build a chatbot.

The client is responsible for maintaining the history of a conversation.
The conversation is divided into messages, each having text content, and
a role name that describes the speaker of that text. Each time a request
is made to the API, the full conversation history is provided, and the
server will generate content for the next message in its history, as the
role of the assistant.

This endpoint supports the following features:

1. Automatic context window caching across requests
2. Streaming responses to give you generated tokens in real time
3. Image uploads via data: URIs

## Request URIs

- `/v1/chat/completions` (OpenAI API compatible)

## Request Methods

- `POST`

## Request Content Types

- `application/json` must be used.

## Request Parameters

- `model`: `string`
  
  Specifies name of model to run.
  
  Only a single model is currently supported, so this field is simply
  copied along to the response. In the future, this will matter.
  
  This field is required in the request.

- `messages`: `array<object<role:string, content:string>>`

  Specifies chat messages.
  
  This is your prompt, except structured as JSON that indicates who's
  saying what. It's a required array containing objects. Each object has
  two mandatory fields:
  
  - `role`: `string`
    
    The role may be any of the following:
    
    - `system`: Used for system prompts.
    - `user`: Used when the user is speaking.
    - `assistant`: Used when the model is speaking.
  
  - `content`: `string`
    
    The content field holds text the system, user, or assistant said. It
    is recommended that this be treated as markdown. This field may
    contain images which should be embedded as data: URIs.
  
  The completion endpoint will then generate at least one additional
  chat message. If the client wishes to request another completion, then
  the client should append the server-provided message to the messages
  field, and then start a new chat completions request over. The server
  will then look at the full conversation history, notice that it's
  cached, and then re-use the same context if it hasn't expired.
  Otherwise, it'll prefill the entire provided messages history into a
  new context to resume a fresh conversation where you left off.

- `stream`: `boolean|null`
  
  If this field is optionally set to true, then this endpoint will
  return a text/event-stream using HTTP chunked transfer encoding. This
  allows your chatbot to rapidly show text as it's being genearted. The
  standard JSON response is slightly modified so that its message field
  will be named delta instead. It's assumed the client will reconstruct
  the full conversation.

- `stream_options`: `object|null`

  Options for streaming the API response. This parameter is only
  applicable when `stream: true` is also specified. Default is `null`.

  - `include_usage`: `boolean|null`

    Whether to include usage statistics in the streaming response. Default is `false`.

    If set to `true`, a `usage` field with the usage information will be
    included in an additional empty chunk. Note that all other chunks will
    also contain this field, but with `null` value.

- `max_tokens`: `integer|null`

  Specifies an upper bound for the number of tokens that can be
  generated for this completion. This can be used to control compute
  and/or latency costs.

- `max_completion_tokens`: `integer|null`

  This currently means the same thing as `max_tokens`.

- `top_p`: `number|null`
  
  May optionally be used to set the `top_p` sampling parameter. This
  should be a floating point number. Setting this to 1.0 (the default)
  will disable this feature. Setting this to, for example, 0.1, would
  mean that only the top 10% probability tokens are considered.
  
  We generally recommend altering this or temperature but not both.

- `temperature`: `number|null`
  
  Configures the randomness level of generated text.
  
  This field may be set to a value between 0.0 and 2.0 inclusive. It
  defaults to 1.0. Lower numbers are more deterministic. Higher numbers
  mean more randomness.
  
  We generally recommend altering this or top_p but not both.

- `seed`: `integer|null`
  
  If specified, llamafiler will make its best effort to sample
  deterministically, even when temperature is non-zero. This means that
  repeated requests with the same seed and parameters should return the
  same result.

- `presence_penalty`: `number|null`
  
  Number between -2.0 and 2.0. Positive values penalize new tokens based
  on whether they appear in the text so far, increasing the model's
  likelihood to talk about new topics.

- `frequency_penalty`: `number|null`
  
  Number between -2.0 and 2.0. Positive values penalize new tokens based
  on their existing frequency in the text so far, decreasing the model's
  likelihood to repeat the same line verbatim.

- `user`: `string|null`
  
  A unique identifier representing your end-user, which can help
  llamafiler to monitor and detect abuse.

- `stop`: `string|array<string>|null`
  
  Up to 4 sequences where the API will stop generating further tokens.

- `response_format`: `string|object|null`
  
  Specifies the format that the model must output.
  
  The default value is `"auto"` which means to just output arbitrary
  text without any kind of grammar constraint. No other string values
  are currently supported.
  
  This field may be set to `{ "type": "json_object" }` to indicate that
  the response must be a valid JSON object, e.g. `{}`, `{"hi": 123}`. If
  you use this feature, then you must have your user in the last chat
  message instruct the LLM to generate JSON, otherwise you might get a
  strange response that never ends.
  
  It's also possible to further constrain the JSON, by specifying a JSON
  schema, by using `{ "type": "json_schema", "json_schema": {...} }`. In
  this case, you're able to specify things like mandatory fields and
  their types.

The following OpenAI Chat Completions request parameters are currently
unsupported:

- `n`
- `tools`
- `audio`
- `logprobs`
- `functions`
- `modalities`
- `tool_choice`
- `top_logprobs`
- `function_call`
- `parallel_tool_calls`

## Context Caching

It may seem surprising that the entire conversation history should be
uploaded each time a new message from the assistant is desired. This is
fast because the server has a fixed number of slots which are used for
caching context windows. When an HTTP client sends a chat completion
request, a prefix search is made for a slot that best matches the
supplied conversation history. Once a slot is chosen, unrelated content
(possibly from another user's conversation) is deleted, the common
prefix is preserved, and the remaining portions are prefilled. If all
slots are in use, then the server handler waits for one to be free.

## Image Uploads

If a vision model was specified by passing the `--mmproj` flag, then
each slot will additionally maintain a vision model context, which is
used to encode embeddings for any data: URIs holding valid images.

The data URI must conform to RFC2397. For example, a 1x1 transparent
pixel could be encoded as:

    data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==

Images can be included by any role. Multiple images may be embedded
within a single message content. The server will evaluate images in the
same order as any surrounding text, e.g.

```markdown
![1x1.gif](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
<img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==">
```

The declared mime type of the data URI must begin with `image/`. Its
subtype is largely ignored, since the file type is inferred from the
binary payload.

The supported image file formats are JPEG, PNG, and GIF.

## See Also

- [LLaMAfiler Documentation Index](index.md)
- [LLaMAfiler Endpoints Reference](endpoints.md)
- [LLaMAfiler Technical Details](technical_details.md)
