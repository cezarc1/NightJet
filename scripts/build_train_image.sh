#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_TAG:?Set IMAGE_TAG to an immutable tag, for example: 0.1.0-$(git rev-parse --short HEAD)}"

IMAGE_REPO="${IMAGE_REPO:-ghcr.io/cezarc1/nightjet-train}"
KUBETORCH_CONTEXT="${KUBETORCH_CONTEXT:-/Users/cezar/Code/kubetorch}"
PLATFORM="${PLATFORM:-linux/amd64}"

docker buildx build \
  --platform "${PLATFORM}" \
  --build-context "kubetorch=${KUBETORCH_CONTEXT}" \
  -f docker/Dockerfile.train \
  -t "${IMAGE_REPO}:${IMAGE_TAG}" \
  --push \
  .

echo "${IMAGE_REPO}:${IMAGE_TAG}"
