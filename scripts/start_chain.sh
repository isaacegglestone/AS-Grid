#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
nohup python3 scripts/run_feature_ci_chain.py --start BB > logs/chain_run.log 2>&1 &
echo "Chain started — PID=$!"
echo "Tail logs with: tail -f logs/chain_run.log"
