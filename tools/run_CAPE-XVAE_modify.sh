#!/bin/bash

if [ "$DOMAIN" == "" ] || [ "$MHC_Is" == "" ] || [ "$XVAE_CKPT_ID" == "" ]; then
  echo "Please set the environmental variables: 'DOMAIN', 'MHC_Is', 'XVAE_CKPT_ID'"
else
  if [ $# -ne 2 ]; then
     echo "run_CAPE-XVAE_modify <profile> <nr of repeats>"
  else
    for ((i=1; i<=${2}; i++))
    do
      cape-xvae.py --domain $DOMAIN --task modify --PROFILE \"${1}\" --SEQ \"natural\" --SEQ_ENC_STOCHASTIC True --MHCs \"$MHC_Is\" --CKPT_ID \"${XVAE_CKPT_ID}\" --MODPARAMS \"modparams\"
    done
  fi
fi
