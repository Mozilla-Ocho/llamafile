prefix=dist

$(prefix):
	mkdir -p $@
	echo "*" > $(prefix)/.gitignore

MODELS_DIR=$(prefix)/.models

$(MODELS_DIR): $(prefix)
	mkdir -p $@

.PHONY: models all

EMBEDFILE=./o/llama.cpp/embedfile/embedfile

MODEL_MXBAI=mxbai-embed-xsmall-v1-f16
MODEL_SNOWFLAKE=snowflake-arctic-embed-m-v1.5-f16
MODEL_NOMIC=nomic-embed-text-v1.5.f16
MODEL_ALLMINI=all-MiniLM-L6-v2.f16

$(MODELS_DIR)/$(MODEL_MXBAI).gguf: $(MODELS_DIR)
	curl -L -o $@ 'https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/resolve/main/gguf/mxbai-embed-xsmall-v1-f16.gguf'

$(MODELS_DIR)/$(MODEL_SNOWFLAKE).gguf: $(MODELS_DIR)
	curl -L -o $@ 'https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/resolve/main/gguf/snowflake-arctic-embed-m-v1.5-f16.gguf'

$(MODELS_DIR)/$(MODEL_NOMIC).gguf: $(MODELS_DIR)
	curl -L -o $@ 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf'

$(MODELS_DIR)/$(MODEL_ALLMINI).gguf: $(MODELS_DIR)
	curl -L -o $@ 'https://huggingface.co/asg017/sqlite-lembed-model-examples/resolve/main/all-MiniLM-L6-v2/all-MiniLM-L6-v2.e4ce9877.f16.gguf'

models: \
	$(MODELS_DIR)/$(MODEL_MXBAI).gguf \
	$(MODELS_DIR)/$(MODEL_SNOWFLAKE).gguf \
	$(MODELS_DIR)/$(MODEL_NOMIC).gguf \
	$(MODELS_DIR)/$(MODEL_ALLMINI).gguf

dist/$(MODEL_MXBAI).embedfile: $(MODELS_DIR)/$(MODEL_MXBAI).gguf $(EMBEDFILE) embedfile.mk
	cp $(EMBEDFILE) $@
	echo "-m\n$(MODEL_MXBAI).gguf\n..." > .args
	./o/llamafile/zipalign -j0 $@ $< .args
	rm .args

dist/$(MODEL_SNOWFLAKE).embedfile: $(MODELS_DIR)/$(MODEL_SNOWFLAKE).gguf $(EMBEDFILE) embedfile.mk
	cp $(EMBEDFILE) $@
	echo "-m\n$(MODEL_SNOWFLAKE).gguf\n..." > .args
	./o/llamafile/zipalign -j0 $@ $< .args
	rm .args

dist/$(MODEL_NOMIC).embedfile: $(MODELS_DIR)/$(MODEL_NOMIC).gguf $(EMBEDFILE) embedfile.mk
	cp $(EMBEDFILE) $@
	echo "-m\n$(MODEL_NOMIC).gguf\n..." > .args
	./o/llamafile/zipalign -j0 $@ $< .args
	rm .args

dist/$(MODEL_ALLMINI).embedfile: $(MODELS_DIR)/$(MODEL_ALLMINI).gguf $(EMBEDFILE) embedfile.mk
	cp $(EMBEDFILE) $@
	echo "-m\n$(MODEL_ALLMINI).gguf\n..." > .args
	./o/llamafile/zipalign -j0 $@ $< .args
	rm .args

all: \
	dist/$(MODEL_MXBAI).embedfile \
	dist/$(MODEL_SNOWFLAKE).embedfile \
	dist/$(MODEL_NOMIC).embedfile \
	dist/$(MODEL_ALLMINI).embedfile
