#!/usr/bin/env bash
export LD_LIBRARY_PATH="/home/jxy/miniconda3/envs/waihai/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/jxy/miniconda3/envs/waihai/bin/python "$@"
