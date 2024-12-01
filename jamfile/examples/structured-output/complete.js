/// <reference types="./embedfile/js_builtins/types.d.ts"/>

import { yellow } from "jamfile:color";
import {CompletionModel} from "jamfile:completionmodel";
import { formatDuration } from "jamfile:fmt";

const model = new CompletionModel('Llama-3.2-1B-Instruct-f16.gguf');

console.log(1);

const article = `Bringing Android to XR
Today, we're introducing Android XR, a new operating system built for this next generation of computing. Created in collaboration with Samsung, Android XR combines years of investment in AI, AR and VR to bring helpful experiences to headsets and glasses.

We’re working to create a vibrant ecosystem of developers and device makers for Android XR, building on the foundation that brought Android to billions. Today’s release is a preview for developers, and by supporting tools like ARCore, Android Studio, Jetpack Compose, Unity, and OpenXR from the beginning, developers can easily start building apps and games for upcoming Android XR devices. For Qualcomm partners like Lynx, Sony and XREAL, we are opening a path for the development of a wide array of Android XR devices to meet the diverse needs of people and businesses. And, we are continuing to collaborate with Magic Leap on XR technology and future products with AR and AI.`;



//console.log(model.complete(`describe qualities about the county of Canada in JSON format.`));

import { z, zodToJsonSchema } from "jamfile:zod";
const Country = z.object({
  name: z.string(),
  capital: z.string(),
  languages: z.array(z.string()),
});

console.log('='.repeat(100));

let prompt = `describe the country of Canada in JSON format, maximum of 3 elements per array: `

let t0;


t0 = Date.now();
console.log(
  model.complete(
    prompt,
    {schema: zodToJsonSchema(Country)}
  )
);
console.log(formatDuration(Date.now() - t0, {ignoreZero: true}))

t0 = Date.now();
console.log(
  model.complete(
    "Describe the person in this sentence: John Doe is 30 years old.",
    {schema: zodToJsonSchema(z.object({
      first_name: z.string(),
      last_name: z.string(),
      age_in_years: z.number(),
    }))}
  )
);
console.log(formatDuration(Date.now() - t0, {ignoreZero: true}))


console.log(
  model.complete("What federal employees are mentioned in the following passage? The long-awaited report from DOJ Inspector General Michael Horowitz's office comes nearly four years after a crowd of Donald Trump's supporters stormed the U.S. Capitol on Jan. 6, 2021, to try to prevent Congress from certifying President Biden's election win. The report looks at how the FBI handled its intelligence and informants, known as confidential human sources, ahead of the event.",
  {
    schema: zodToJsonSchema(
      z.object({
        name: z.string(),
        occupation: z.string(),
        organization: z.string(),
      })
    )
  })
)

console.log(
  model.complete(`

    What questions does the following passage answer? Respond with a list of question/answer pairs. Provide only relevant question that this passage could answer, not generic ones.

    EXAMPLE: [{\"question\": \"Who delivers presents to children every Christmas?\", \"answer\": \"Santa Clause.\"}]

    PASSAGE: The long-awaited report from DOJ Inspector General Michael Horowitz's office comes nearly four years after a crowd of Donald Trump's supporters stormed the U.S. Capitol on Jan. 6, 2021, to try to prevent Congress from certifying President Biden's election win. The report looks at how the FBI handled its intelligence and informants, known as confidential human sources, ahead of the event.
    `,
  {
    schema: zodToJsonSchema(
      z.array(z.object({
        question: z.string(),
        answer: z.string()
      }))
    )
  })
)



const mathReasoningSchema = z.object({
  steps: z.array(
    z.object({
      explanation: z.string(),
      output: z.string(),
    })
  ),
  final_answer: z.string(),
}).strict();

prompt = `
    You are a helpful math tutor. You will be provided with a math problem,
    and your goal will be to output a step by step solution, along with a final answer.
    For each step, just provide the output as an equation use the explanation field to detail the reasoning.

    PROMPT: how can I solve 8x + 7 = -23
`;
const result = model.complete(prompt, {schema: zodToJsonSchema(mathReasoningSchema)});

console.log(yellow(prompt));
console.log(result);
