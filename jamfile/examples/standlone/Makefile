Llama-3.2-1B-Instruct-Q4_K_M.gguf:
	curl -L -o $@ https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/$@

mxbai-embed-xsmall-v1-f16.gguf:
	curl -L -o $@ https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/resolve/main/gguf/$@

standalone.jamfile:  \
	../../../o/jamfile/jamfile \
	Llama-3.2-1B-Instruct-Q4_K_M.gguf \
	mxbai-embed-xsmall-v1-f16.gguf \
	standalone.js 
	
	printf "%s\n" \
		"--default-embedding-model" \
		"mxbai-embed-xsmall-v1-f16.gguf" \
		"--default-completion-model" \
		"Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
		"run" \
		"standalone.js" \
		"..." > .args
	
	cp ../../../o/jamfile/jamfile $@
	../../../o/llamafile/zipalign -j0 $@ $^ .args
	rm .args
	chmod u+x $@
