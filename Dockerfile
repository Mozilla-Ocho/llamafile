ARG RELEASE_TAG

FROM ubuntu:latest AS downloader

WORKDIR /download

RUN apt-get update && apt-get install -y curl

RUN curl -L -o ./llamafile-server https://github.com/Mozilla-Ocho/llamafile/releases/download/${RELEASE_TAG}/llamafile-server-${RELEASE_TAG}

RUN chmod +x ./llamafile-server



FROM scratch AS final

RUN addgroup --gid 1000 user
RUN adduser --uid 1000 --gid 1000 --disabled-password --gecos "" user

# Switch to user.
USER user

WORKDIR /usr/local/bin

COPY --from=downloader /download/llamafile-server ./llamafile-server

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llamafile-server", "--host", "0.0.0.0"]