#!/usr/bin/env bash
shuf -n 1000000 training.tsv | training1m.tsv.gz
