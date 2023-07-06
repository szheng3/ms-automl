FROM mcr.microsoft.com/devcontainers/python:3.10-buster
#FROM rust:latest

# Include lld linker to improve build times either by using environment variable
# RUSTFLAGS="-C link-arg=-fuse-ld=lld" or with Cargo's configuration file (i.e see .cargo/config.toml).
#RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#   && apt-get -y install clang lld \
#   && apt-get autoremove -y && apt-get clean -y

RUN apt-get update && \
    apt-get install -y ffmpeg



# Add the PyTorch repository
#RUN add-apt-repository ppa:ubuntu-toolchain-r/test

# Update the package repository and install PyTorch
#RUN apt-get update && \
#    apt-get install -y python3-torch

#RUN apt-get update && \
#    apt-get install docker.io -y