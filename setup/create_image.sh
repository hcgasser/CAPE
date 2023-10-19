#!/bin/bash

read -p 'Set container password (e.g. cape_pwd): ' password

pushd $CAPE
docker build --build-arg REPO=CAPE --build-arg PASSWORD=${password} -f setup/Dockerfile -t cape .
popd
