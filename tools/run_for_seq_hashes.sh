#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
  echo 'The template will be called for each seq_hash in the seq_hash_file_path'
  echo 'Every #SEQ_HASH# in the template is replaced by the seq hashes'
  echo "Usage: $0 <template> <seq_hash_file_path>"
else

    template="$1"
    seq_hash_file_path="$2"

    if ! [ -e "$seq_hash_file_path" ]; then
      echo "file not found: $seq_hash_file_path"
    else
        while IFS= read -r seq_hash; do
          command="${template//\#SEQ_HASH#/$seq_hash}"
          
          eval "$command"
        done < "$seq_hash_file_path"
    fi
fi
