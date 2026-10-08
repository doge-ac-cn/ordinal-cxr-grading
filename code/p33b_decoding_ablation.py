"""P3.3b official checkpoint decoding ablation: chained vs counting-rule decoding.
One pass over the test set; per-image predictions for both decodings are saved so that
paired patient-level analyses can be reproduced.
Outputs results/p33b_decoding_ablation.json + results/p33_official_preds.npz."""
import json
import time
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJ / "code"))
import p33_official_compare as p33  # noqa: E402
from p33_official_compare import load_batches, OFFICIAL_COLS, N_LEVELS, OUR_COLS  # noqa: E402,F401


def main():
    from cxr.models.mst import MSTRegression
    t0 = time.time()
    ckpt = torch.load(PROJ / "official-ckpt" / "ordinal.ckpt", map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    out_dim = [v.shape for k, v in sd.items() if k.endswith("linear.weight")]
    model = MSTRegression(in_ch=1, out_ch=int(out_dim[0][0]), task="ordinal",
                          loss_kwargs={"class_labels_num": [N_LEVELS[c] - 1 for c in OFFICIAL_COLS]})
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()

    chained, counting, uids_all = {c: [] for c in OFFICIAL_COLS}, {c: [] for c in OFFICIAL_COLS}, []
    n_done = 0
    with torch.no_grad():
        for x, Y, us in load_batches("test"):
            logit = model(x.cuda()).float().cpu()
            off = 0
            for c in OFFICIAL_COLS:
                k = N_LEVELS[c] - 1
                blk = logit[:, off:off + k]
                sig = torch.sigmoid(blk)
                cum = torch.cumprod(sig, dim=1)  # joint cumulative P(y > k)
                chained[c].append((cum > 0.5).sum(1))
                counting[c].append((sig > 0.5).sum(1))  # raw conditionals, underestimates QWK
                off += k
            uids_all += us
            n_done += len(x)
            print(f"  {n_done} ({time.time()-t0:.0f}s)", flush=True)

    from sklearn.metrics import cohen_kappa_score
    Y = {c: [] for c in OFFICIAL_COLS}
    for _, Yb, _ in load_batches("test"):
        for c in OFFICIAL_COLS:
            Y[c].append(Yb[:, OUR_COLS.index(c)])
    res = {"model": "official ordinal.ckpt (DINOv2 ViT-S/14 + linear CORN head)",
           "n": n_done, "decodings": {}}
    for name, P in (("chained", chained), ("counting_rule", counting)):
        pf, q = {}, {}
        for ci, c in enumerate(OFFICIAL_COLS):
            p = torch.cat(P[c]).numpy() if isinstance(P[c][0], torch.Tensor) else np.concatenate(P[c])
            y = torch.cat(Y[c]).numpy()
            k = N_LEVELS[c]
            pf[c] = {"qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(k)))), 4)}
            q[c] = pf[c]["qwk"]
        res["decodings"][name] = {"per_task_qwk": pf, "mean_qwk": round(float(np.mean(list(q.values()))), 4)}
    res["delta_mean_qwk"] = round(res["decodings"]["chained"]["mean_qwk"] - res["decodings"]["counting_rule"]["mean_qwk"], 4)
    (PROJ / "results" / "p33b_decoding_ablation.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(PROJ / "results" / "p33_official_preds.npz",
                        uid=np.array(uids_all),
                        **{f"{c}/chained": torch.cat(chained[c]).numpy() for c in OFFICIAL_COLS},
                        **{f"{c}/counting": torch.cat(counting[c]).numpy() for c in OFFICIAL_COLS})
    print("chained mean:", res["decodings"]["chained"]["mean_qwk"],
          "| counting mean:", res["decodings"]["counting_rule"]["mean_qwk"],
          "| delta:", res["delta_mean_qwk"], flush=True)
    print("wrote results/p33b_decoding_ablation.json + p33_official_preds.npz", flush=True)


if __name__ == "__main__":
    main()
