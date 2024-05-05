#!/bin/bash

rm -f $KSLICER_CFG
cat >> $KSLICER_CFG << EOF
$1
$2
EOF
