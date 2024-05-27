
###############################################################################
FROM alpine as build

RUN apk update && \
    apk add make

WORKDIR /build
COPY . .
RUN make -j $(nproc)

###############################################################################
FROM alpine as package

WORKDIR /package

COPY --from=build /build/o/llama.cpp/main/main              ./llamafile
COPY --from=build /build/o/llamafile/zipalign               ./zipalign
COPY --from=build /build/models/TinyLLama-v0.1-5M-F16.gguf  ./model.gguf

RUN echo "-m" > .args
RUN echo "model.gguf" >> .args
RUN ./zipalign -j0 llamafile model.gguf .args

RUN chmod +x ./llamafile

###############################################################################
FROM alpine as final

WORKDIR /usr/src/app

COPY --from=package /package/llamafile ./llamafile

RUN ./llamafile -e -p '## Famous Speech\n\nFour score and seven' -n 50 -ngl 0

ENTRYPOINT ["/bin/sh", "/usr/src/app/llamafile"]
CMD ["--cli", "-p", "hello world the gruff man said"]

## Just running test?
# docker build -f .devops/ci.Dockerfile -t llamafile_ci .
# docker run llamafile_ci

## Get the binary?
# docker build -f .devops/ci.Dockerfile -t llamafile_ci .
# docker create --name llamafile_ci_container llamafile_ci
# docker cp llamafile_ci_container:/usr/src/app/llamafile test.llamafile
