#! /bin/bash

ROOT=$(dirname $(realpath $0))

set -e

TF_URL="https://github.com/tensorflow/tensorflow"

mkdir -p "${ROOT}/third_party"
if ! [[ -d "${ROOT}/third_party/tensorflow" ]]; then
    git clone --depth=1 --recurse-submodules --shallow-submodules "$TF_URL" "${ROOT}/third_party/tensorflow"
fi
