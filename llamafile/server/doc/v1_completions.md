# LLaMAfiler Completions Endpoint

The `/v1/completions` endpoint generates text completions based on a
given prompt. It provides a flexible interface for text generation,
allowing customization of parameters such as temperature, top-p
sampling, and maximum tokens.

This endpoint supports the following features:

1. Deterministic outputs using a fixed seed
2. Streaming responses for real-time token generation
3. Configurable stopping criteria for token generation

## Request URIs

- `/v1/completions` (OpenAI API compatible)

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

- `prompt`: `string`
  
  The input text that the model will generate a completion for.
  
  This field is required.

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
  
  Specifies up to 4 stop sequences where the API will cease text generation.

## See Also

- [LLaMAfiler Documentation Index](index.md)
- [LLaMAfiler Endpoints Reference](endpoints.md)
- [LLaMAfiler Technical Details](technical_details.md)
