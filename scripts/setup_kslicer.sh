#!/bin/bash
rm -f kslicer_cfg.txt
cat >> kslicer_cfg.txt << EOF
$1
$2
EOF
