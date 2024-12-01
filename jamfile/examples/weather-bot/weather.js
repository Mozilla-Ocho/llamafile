/**
 * A simple bot that takes in weather info from wttr.in and tries to respond
 * in helpful Engish. Big work in progress!
 * 
 * run with:
 * 
 * ```bash
 * jamfile run weather.js
 * ```
 * 
 * Requires curl and a llama 3.2 GGUF - run `make all` to download it.
 * 
 * TODO:
 * - Edit prompt more to generate more helpful responses
 * - Include more weather data in prompt (hourly forecast for the day, etc.)
 * - Only include useful weather data, not humidity, 
 * - Save more user preferences - available wardrobe, temperature cut-offs, etc
 * - Move away from popen() + curl
 */

/// <reference types="../../jamfile.d.ts"/>

import { CompletionModel } from "jamfile:llamafile";
import { popen } from "qjs:std";
import wttrSchema from "./wttr-schema.js";

const MODEL_PATH = './Llama-3.2-1B-Instruct-Q4_K_M.gguf';

function fetchWttr() {
  const wttrin = popen("curl -s 'wttr.in?format=j1'", 'r').readAsString();
  return wttrSchema.parse(JSON.parse(wttrin));
  
}

function createPrompt(data) {
  return `<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an assitant who will greet the user like a butler, 
providing information about the weather.

Use the following data from the given JSON blob to provide information and recommendations in English.

User preferences:
  - Only give temperatures in Farenheit
  - Warn if it'll be colder than 60 or warmer than 80
  - Use a british accent
  - Add a couple emojis as well
  - Keep sentences short, use line breaks in between.
    

<|eot_id|><|start_header_id|>user<|end_header_id|>

${JSON.stringify(data.current_condition)} 

<|eot_id|><|start_header_id|>assistant<|end_header_id|>`;
}


function main() {
  const model = new CompletionModel(MODEL_PATH);
  const data = fetchWttr();
  const response = model.complete(createPrompt(data));
  
  console.log(response);
}

if(import.meta.main) main();
