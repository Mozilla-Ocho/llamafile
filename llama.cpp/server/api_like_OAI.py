#!/usr/bin/env python3
import argparse
from flask import Flask, jsonify, request, Response
import urllib.parse
import requests
import time
import json


app = Flask(__name__)
slot_id = -1

parser = argparse.ArgumentParser(description="An example of using server.cpp with a similar API to OAI. It must be used together with server.cpp.")
parser.add_argument("--chat-prompt", type=str, help="the top prompt in chat completions(default: 'A chat between a curious user and an artificial intelligence assistant. The assistant follows the given rules no matter what.\\n')", default='A chat between a curious user and an artificial intelligence assistant. The assistant follows the given rules no matter what.\\n')
parser.add_argument("--user-name", type=str, help="USER name in chat completions(default: '\\nUSER: ')", default="\\nUSER: ")
parser.add_argument("--ai-name", type=str, help="ASSISTANT name in chat completions(default: '\\nASSISTANT: ')", default="\\nASSISTANT: ")
parser.add_argument("--system-name", type=str, help="SYSTEM name in chat completions(default: '\\nASSISTANT's RULE: ')", default="\\nASSISTANT's RULE: ")
parser.add_argument("--stop", type=str, help="the end of response in chat completions(default: '</s>')", default="</s>")
parser.add_argument("--llama-api", type=str, help="Set the address of server.cpp in llama.cpp(default: http://127.0.0.1:8080)", default='http://127.0.0.1:8080')
parser.add_argument("--api-key", type=str, help="Set the api key to allow only few user(default: NULL)", default="")
parser.add_argument("--host", type=str, help="Set the ip address to listen.(default: 127.0.0.1)", default='127.0.0.1')
parser.add_argument("--port", type=int, help="Set the port to listen.(default: 8081)", default=8081)

args = parser.parse_args()

def is_present(json, key):
    try:
        buf = json[key]
    except KeyError:
        return False
    if json[key] == None:
        return False
    return True

#convert chat to prompt
def convert_chat(messages):
    prompt = "" + args.chat_prompt.replace("\\n", "\n")
    image_data = []
    image_id = 1

    system_n = args.system_name.replace("\\n", "\n")
    user_n = args.user_name.replace("\\n", "\n")
    ai_n = args.ai_name.replace("\\n", "\n")
    stop = args.stop.replace("\\n", "\n")


    for line in messages:
        if (line["role"] == "system"):
            prompt += f"{system_n}{line['content']}"
        if (line["role"] == "user"):
            # content can either be a string or an iterable with "text"
            # and "image_url" elements
            content = line['content']
            if type(content) == str:
                prompt += f"{user_n}{line['content']}"
            else:
                # add all elements from array
                for content_part in content:
                    if content_part['type'] == "text":
                        prompt += f"{user_n}{content_part['text']}{stop}"
                    elif content_part['type'] == "image_url":
                        image_data.append(
                            {"data": content_part['image_url']['url'].split(",")[1],
                             "id": image_id})
                        image_id+=1

        if (line["role"] == "assistant"):
            prompt += f"{ai_n}{line['content']}{stop}"
    prompt += ai_n.rstrip()

    return prompt, image_data

def make_postData(body, chat=False, stream=False):
    postData = {}
    if (chat):
        prompt, image_data = convert_chat(body["messages"])
        postData["prompt"] = prompt
        if len(image_data) > 0:
            postData["image_data"] = image_data
    else:
        postData["prompt"] = body["prompt"]
    if(is_present(body, "temperature")): postData["temperature"] = body["temperature"]
    if(is_present(body, "top_k")): postData["top_k"] = body["top_k"]
    if(is_present(body, "top_p")): postData["top_p"] = body["top_p"]
    if(is_present(body, "max_tokens")): postData["n_predict"] = body["max_tokens"]
    if(is_present(body, "presence_penalty")): postData["presence_penalty"] = body["presence_penalty"]
    if(is_present(body, "frequency_penalty")): postData["frequency_penalty"] = body["frequency_penalty"]
    if(is_present(body, "repeat_penalty")): postData["repeat_penalty"] = body["repeat_penalty"]
    if(is_present(body, "mirostat")): postData["mirostat"] = body["mirostat"]
    if(is_present(body, "mirostat_tau")): postData["mirostat_tau"] = body["mirostat_tau"]
    if(is_present(body, "mirostat_eta")): postData["mirostat_eta"] = body["mirostat_eta"]
    if(is_present(body, "seed")): postData["seed"] = body["seed"]
    if(is_present(body, "logit_bias")): postData["logit_bias"] = [[int(token), body["logit_bias"][token]] for token in body["logit_bias"].keys()]
    if (args.stop != ""):
        postData["stop"] = [args.stop]
    else:
        postData["stop"] = []
    if(is_present(body, "stop")): postData["stop"] += body["stop"]
    postData["n_keep"] = -1
    postData["stream"] = stream
    postData["cache_prompt"] = True
    postData["slot_id"] = slot_id
    return postData

def make_resData(data, chat=False, promptToken=[]):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion" if (chat) else "text_completion",
        "created": int(time.time()),
        "truncated": data["truncated"],
        "model": "LLaMA_CPP",
        "usage": {
            "prompt_tokens": data["tokens_evaluated"],
            "completion_tokens": data["tokens_predicted"],
            "total_tokens": data["tokens_evaluated"] + data["tokens_predicted"]
        }
    }
    if (len(promptToken) != 0):
        resData["promptToken"] = promptToken
    if (chat):
        #only one choice is supported
        resData["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": data["content"],
            },
            "finish_reason": "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
        }]
    else:
        #only one choice is supported
        resData["choices"] = [{
            "text": data["content"],
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
        }]
    return resData

def make_resData_stream(data, chat=False, time_now = 0, start=False):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion.chunk" if (chat) else "text_completion.chunk",
        "created": time_now,
        "model": "LLaMA_CPP",
        "choices": [
            {
                "finish_reason": None,
                "index": 0
            }
        ]
    }
    slot_id = data["slot_id"]
    if (chat):
        if (start):
            resData["choices"][0]["delta"] =  {
                "role": "assistant"
            }
        else:
            resData["choices"][0]["delta"] =  {
                "content": data["content"]
            }
            if (data["stop"]):
                resData["choices"][0]["finish_reason"] = "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
    else:
        resData["choices"][0]["text"] = data["content"]
        if (data["stop"]):
            resData["choices"][0]["finish_reason"] = "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"

    return resData


@app.route('/chat/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    if (args.api_key != "" and request.headers["Authorization"].split()[1] != args.api_key):
        return Response(status=403)
    body = request.get_json()
    stream = False
    tokenize = False
    if(is_present(body, "stream")): stream = body["stream"]
    if(is_present(body, "tokenize")): tokenize = body["tokenize"]
    postData = make_postData(body, chat=True, stream=stream)

    promptToken = []
    if (tokenize):
        tokenData = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/tokenize"), data=json.dumps({"content": postData["prompt"]})).json()
        promptToken = tokenData["tokens"]

    if (not stream):
        data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData))
        print(data.json())
        resData = make_resData(data.json(), chat=True, promptToken=promptToken)
        return jsonify(resData)
    else:
        def generate():
            data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData), stream=True)
            time_now = int(time.time())
            resData = make_resData_stream({}, chat=True, time_now=time_now, start=True)
            yield 'data: {}\n'.format(json.dumps(resData))
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    resData = make_resData_stream(json.loads(decoded_line[6:]), chat=True, time_now=time_now)
                    yield 'data: {}\n'.format(json.dumps(resData))
        return Response(generate(), mimetype='text/event-stream')


@app.route('/completions', methods=['POST'])
@app.route('/v1/completions', methods=['POST'])
def completion():
    if (args.api_key != "" and request.headers["Authorization"].split()[1] != args.api_key):
        return Response(status=403)
    body = request.get_json()
    stream = False
    tokenize = False
    if(is_present(body, "stream")): stream = body["stream"]
    if(is_present(body, "tokenize")): tokenize = body["tokenize"]
    postData = make_postData(body, chat=False, stream=stream)

    promptToken = []
    if (tokenize):
        tokenData = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/tokenize"), data=json.dumps({"content": postData["prompt"]})).json()
        promptToken = tokenData["tokens"]

    if (not stream):
        data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData))
        print(data.json())
        resData = make_resData(data.json(), chat=False, promptToken=promptToken)
        return jsonify(resData)
    else:
        def generate():
            data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData), stream=True)
            time_now = int(time.time())
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    resData = make_resData_stream(json.loads(decoded_line[6:]), chat=False, time_now=time_now)
                    yield 'data: {}\n'.format(json.dumps(resData))
        return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(args.host, port=args.port)
