#!/bin/bash
cd "$(dirname "$0")"
run() { # tag fdir pool weight seed
  if [ -f "../results/linear_probe_seed$5_$1.json" ]; then echo "skip $1 s$5"; return; fi
  python3 train_ordinal.py --epochs 100 --seed $5 --tag "$1" --features-dir "$2" --pool "$3" $4 > "run_$1_s$5.log" 2>&1 \
    && echo "done $1 s$5" || echo "FAIL $1 s$5"
}
for seed in 0 1 2; do
  run p16_224bc_u features        cls  ""            $seed
  run p16_224bc_w features        cls  "--cls-weight" $seed
  run p16_224bm_u features        mean ""            $seed
  run p16_224bm_w features        mean "--cls-weight" $seed
  run p16_224lc_u features-large  cls  ""            $seed
  run p16_224lc_w features-large  cls  "--cls-weight" $seed
  run p16_224lm_u features-large  mean ""            $seed
  run p16_224lm_w features-large  mean "--cls-weight" $seed
  run p16_518bc_u features-518    cls  ""            $seed
  run p16_518bc_w features-518    cls  "--cls-weight" $seed
  run p16_518bm_u features-518    mean ""            $seed
  run p16_518bm_w features-518    mean "--cls-weight" $seed
done
echo "P1.6 "
