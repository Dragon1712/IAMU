#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${CXX:=g++}"
"$CXX" -std=c++17 -O2 -fPIC -shared -Wall -Wextra \
  BACAP_env.cpp -o libBACAP_env.so
printf '\nBuilt: %s/libBACAP_env.so\n' "$PWD"
file libBACAP_env.so
printf '\nExported C symbols:\n'
nm -D --defined-only libBACAP_env.so | grep -E ' (create_env|delete_env|reset|step)$'
