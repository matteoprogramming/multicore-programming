#!/usr/bin/env bash

N=$1

for f in *; do
  if [[ -f "$f" && -x "$f" && "$f" != *.sh && "$f" != *.py  ]]; then

    echo "===== $f $N ====="

    ./"$f" "$N"

    echo
  fi
done
