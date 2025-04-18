#!/bin/bash
echo "$0 executed on $(hostname)"
ssh -T belenos << EOF
  echo "Now connected to \$(hostname)"
  bash SAVE/request_and_launch.sh "$1"
EOF
