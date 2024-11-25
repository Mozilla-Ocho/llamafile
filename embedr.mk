dist/mxbai-embed-xsmall-v1-f16.embedr: ./o/llama.cpp/embedr/embedr
	cp $< $@

dist/snowflake-arctic-embed-m-v1.5-f16.embedr: ./o/llama.cpp/embedr/embedr embedr.mk
	cp $< $@
	echo "-m\nmodels/snowflake-arctic-embed-m-v1.5-f16.gguf\n..." > .args
	./o/llamafile/zipalign -j0 \
		$@ \
		models/snowflake-arctic-embed-m-v1.5-f16.gguf \
		.args
	rm .args


all: dist/mxbai-embed-xsmall-v1-f16.embedr dist/snowflake-arctic-embed-m-v1.5-f16.embedr
