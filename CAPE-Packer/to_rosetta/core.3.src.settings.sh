#!/bin/bash

# modify core.3.src.settings
sed -i '662a\		"MHCEpitopePredictorClient",
935a\	"json-c"' ${1}

