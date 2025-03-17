# jamfile sample: standalone

An example of how you can make a standlone single-binary script from a Jamfile JavaScript script.

The `standalone.js` is a sample JavaScript script that generates a a sample embedding and completion result. It could be ran as-is with `jamfile run standlone.js`, but we could bundle the JS script and model weights into a single-file `jamfile` that works on all operating systems.

To do so, run the following command:

```
make sample.jamfile
```

That will download an embeddings model (MixedBread xsmall) and a LLM (Lllama 3.2), embedding them into a new `sample.jamfile` file with the `standalone.js` JavaScript script.

You can now execute that `jamfile` directly like so:


```
$ ./standalone.jamfile 
embedding sample:  [-0.05826485529541969,0.04374322667717934,0.03084852732717991,0.047234565019607544,-0.05963338911533356,-0.03614785894751549,0.03814202547073364,0.005149350967258215]
completion sample:  
Here is a single haiku about SpongeBob SquarePants:

Fluffy sponge so bright
Rainbow colors in his heart
Joyful underwater

I hope you like it! Let me know if you have any other requests.
```

The output `standalone.jamfile` is a small `850MB`, which includes the LLM, embeddings model, and `jamfile` executable all in one.

- `jamfile` base: `15MB`
- `mxbai-embed-xsmall-v1-f16.gguf`: `49MB`
- `Llama-3.2-1B-Instruct-Q4_K_M.gguf`: `770MB`
- `standalone.jamfile`: `850MB`