#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$BASE_DIR/../proto"
OUT_DIR="$BASE_DIR/smg_grpc_proto/proto"

mkdir -p "$OUT_DIR"

# Ensure protoc and plugins are installed
if ! command -v protoc &> /dev/null; then
    echo "protoc not found. Please install protobuf-compiler."
    return 1 2>/dev/null || true
fi

if ! command -v protoc-gen-go &> /dev/null; then
    echo "protoc-gen-go not found. Installing..."
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
fi

if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "protoc-gen-go-grpc not found. Installing..."
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

# Need to update paths since go mod was init in base dir
export PATH="$PATH:$(go env GOPATH)/bin"

OPTS=""
for file in "$PROTO_DIR"/*.proto; do
    base=$(basename $file .proto)
    OPTS="$OPTS --go_opt=M$(basename $file)=github.com/lightseekorg/smg/crates/grpc_client/golang/smg_grpc_proto/$base --go-grpc_opt=M$(basename $file)=github.com/lightseekorg/smg/crates/grpc_client/golang/smg_grpc_proto/$base"
done

for file in "$PROTO_DIR"/*.proto; do
    echo "Compiling $file..."
    base=$(basename $file .proto)
    mkdir -p "$BASE_DIR/smg_grpc_proto/$base"
    protoc \
        --proto_path="$PROTO_DIR" \
        --go_out="$BASE_DIR/smg_grpc_proto/$base" --go_opt=paths=source_relative \
        --go-grpc_out="$BASE_DIR/smg_grpc_proto/$base" --go-grpc_opt=paths=source_relative \
        $OPTS \
        "$file"
done

echo "Done compiling go proto definitions."
