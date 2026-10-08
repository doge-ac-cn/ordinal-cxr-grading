"""P50  A:  (GPU, )
E1 2x2: 518bm cumulative {count, argmax(scaled)} vs plain CE {argmax}, 3 seeds,
E2 : valoffsetQWK -> test + recall
E4 MSE: 518bm, 3 seeds (+)
E8 : RAD-DINO 224 CLS, {CE, CORAL-shared, CORN, cumul} x 3 seeds
 results/p50_review_experiments.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p25_main_table import LABEL_COLS, N_LEVELS, load_split, CumHead, predict, qwk_mean

PROJ = Path(__file__).resolve().parents[1]
OUT = {}


def cumulative_targets(y, k):
    t = torch.arange(k - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > t).float()


def train_generic(Xtr, Ytr, Xva, Yva, loss_kind, seed, epochs=100, patience=10, weighted=False):
    """loss_kind: ce | coral_shared | corn | cumul | mse"""
    torch.manual_seed(seed)
    model = CumHead(Xtr.shape[1], LABEL_COLS).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if weighted:
        pos_w = {}
        for ci, c in enumerate(LABEL_COLS):
            y = Ytr[:, ci].cuda()
            pw = []
            for t in range(N_LEVELS[c] - 1):
                pos = (y > t).sum().float().clamp(min=1)
                pw.append((len(Ytr) - pos) / pos)
            pos_w[c] = torch.stack(pw)
    n = len(Xtr)
    best, bad, best_state = -1, 0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            x = Xtr[idx].cuda()
            loss = 0.0
            for ci, c in enumerate(LABEL_COLS):
                z = model(x, c)
                y = Ytr[idx, ci].cuda()
                K = N_LEVELS[c]
                if loss_kind == "ce":
                    loss = loss + F.cross_entropy(z, y)
                elif loss_kind == "mse":
                    loss = loss + F.mse_loss(z.squeeze(-1), y.float())
                elif loss_kind == "cumul":
                    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w[c]) if weighted else nn.BCEWithLogitsLoss()
                    loss = loss + crit(z, cumulative_targets(y, K))
                elif loss_kind == "coral_shared":
                    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w[c]) if weighted else nn.BCEWithLogitsLoss()
                    loss = loss + crit(z, cumulative_targets(y, K))
                elif loss_kind == "corn":
                    tgt = cumulative_targets(y, K)
                    mask = torch.cat([torch.ones_like(y, dtype=torch.bool), y > 0], dim=1)[:, :K - 1]
                    lo = F.binary_cross_entropy_with_logits(z, tgt, reduction="none")
                    m = mask.float()
                    loss = loss + (lo * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        mq = qwk_mean(model, Xva, Yva)
        if mq > best:
            best, bad = mq, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def raw_cum_logits(model, X, bs=16384):
    model.eval()
    out = {c: torch.zeros(len(X), N_LEVELS[c] - 1) for c in LABEL_COLS}
    for i in range(0, len(X), bs):
        x = X[i:i + bs].cuda()
        for c in LABEL_COLS:
            out[c][i:i + len(x)] = torch.sigmoid(model(x, c)).cpu()
    return out


def qwk_from_preds(Y, P):
    return float(np.mean([cohen_kappa_score(Y[:, ci].numpy(), P[c].numpy(), weights="quadratic",
                                            labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]))


def main():
    t0 = time.time()
    (Xtr, Ytr, mtr), (Xva, Yva, _), (Xte, Yte, _) = (load_split("features-518", "mean", s) for s in ("train", "val", "test"))

    e1 = {"count": [], "argmax_scaled": [], "ce_argmax": []}
    pt_gaps = {c: [] for c in LABEL_COLS}
    for seed in (0, 1, 2):
        model, _ = train_generic(Xtr, Ytr, Xva, Yva, "cumul", seed)
        P = raw_cum_logits(model, Xte)
        Pc = {c: (P[c] > 0.5).sum(1) for c in LABEL_COLS}
        Pa = {}
        for c in LABEL_COLS:
            ext = torch.cat([torch.ones(len(P[c]), 1), P[c], torch.zeros(len(P[c]), 1)], dim=1)
            pr = (ext[:, :-1] - ext[:, 1:]).clamp(min=0)
            pr = pr / pr.sum(1, keepdim=True).clamp(min=1e-8)
            Pa[c] = pr.argmax(1)
        e1["count"].append(qwk_from_preds(Yte, Pc))
        e1["argmax_scaled"].append(qwk_from_preds(Yte, Pa))
        for c in LABEL_COLS:
            y = Yte[:, LABEL_COLS.index(c)].numpy()
            k = N_LEVELS[c]
            q_count = cohen_kappa_score(y, Pc[c].numpy(), weights="quadratic", labels=list(range(k)))
            q_arg = cohen_kappa_score(y, Pa[c].numpy(), weights="quadratic", labels=list(range(k)))
            pt_gaps[c].append(float(q_count - q_arg))
        del model
        torch.cuda.empty_cache()
    from train_ordinal import train_one
    for seed in (0, 1, 2):
        model = train_one("ce", Xtr, Ytr, Xva, Yva, epochs=100, seed=seed)[0]
        with torch.no_grad():
            Pc = {c: torch.cat([model(Xte[i:i + 16384].cuda(), c).argmax(1).cpu()
                                for i in range(0, len(Xte), 16384)]) for c in LABEL_COLS}
        e1["ce_argmax"].append(qwk_from_preds(Yte, Pc))
        del model
        torch.cuda.empty_cache()
    e1["per_task_gap_mean"] = {c: round(float(np.mean(pt_gaps[c])), 4) for c in LABEL_COLS}
    OUT["E1_inference_rule_2x2"] = {
        "config": "518bm cumulative, 3 seeds; count=(sig>0.5).sum vs argmax of normalized telescoped probs; CE argmax",
        "mean_qwk": {k: [round(v, 4) for v in vals] for k, vals in e1.items() if isinstance(vals, list)},
        "avg": {k: round(float(np.mean(v)), 4) for k, v in e1.items() if isinstance(v, list)},
        "per_task_count_minus_argmax": e1["per_task_gap_mean"]}
    print("E1 done", json.dumps(OUT["E1_inference_rule_2x2"]["avg"]), flush=True)

    model, _ = train_generic(Xtr, Ytr, Xva, Yva, "cumul", 0)
    Pv, Pt = raw_cum_logits(model, Xva), raw_cum_logits(model, Xte)

    def qwk_with_offsets(P, Y, offs):
        Pp = {c: (P[c] > 0.5 + offs[ci]).sum(1) for ci, c in enumerate(LABEL_COLS)}
        return qwk_from_preds(Y, Pp)

    grid = [x / 100 for x in range(-30, 31, 5)]
    best_off, best_v = None, -1
    for o in grid:
        v = qwk_with_offsets(Pv, Yva, [o] * 8)
        if v > best_v:
            best_v, best_off = v, o
    e2 = {"val_best_offset": best_off, "val_qwk": round(best_v, 4),
          "test_qwk_at_0.5": round(qwk_with_offsets(Pt, Yte, [0.0] * 8), 4),
          "test_qwk_at_val_best": round(qwk_with_offsets(Pt, Yte, [best_off] * 8), 4)}
    def grade_recalls(P, Y, off):
        rec = {}
        for ci, c in enumerate(LABEL_COLS):
            pred = (P[c] > 0.5 + off).sum(1).numpy()
            y = Y[:, ci].numpy()
            top = N_LEVELS[c] - 1
            rec[c] = round(float((pred[y == top] == top).mean()) if (y == top).sum() else 0.0, 4)
        return rec
    e2["top_grade_recall_off0.5"] = grade_recalls(Pt, Yte, 0.0)
    e2["top_grade_recall_valbest"] = grade_recalls(Pt, Yte, best_off)
    pred05 = (Pt["HeartSize"] > 0.5).sum(1).numpy()
    predvb = (Pt["HeartSize"] > 0.5 + best_off).sum(1).numpy()
    yh = Yte[:, 0].numpy()
    e2["HeartSize_g3_support"] = int((yh == 3).sum())
    e2["HeartSize_g3_recall_off0.5"] = round(float((pred05[yh == 3] == 3).mean()), 4)
    e2["HeartSize_g3_recall_valbest"] = round(float((predvb[yh == 3] == 3).mean()), 4)
    OUT["E2_threshold_sweep"] = e2
    print("E2 done", json.dumps(e2), flush=True)
    del model, Pv, Pt
    torch.cuda.empty_cache()

    class RegHead(nn.Module):
        def __init__(self, dim, cols):
            super().__init__()
            self.cols = cols
            self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c]) for c in cols})

        def forward(self, x, col):
            return self.heads[col](x)

    patience_default = 10

    def preds_of(model, X):
        with torch.no_grad():
            return {c: torch.cat([model(X[i:i + 16384].cuda(), c) for i in range(0, len(X), 16384)]).cpu().argmax(1)
                    for c in LABEL_COLS}

    def train_mse(seed):
        torch.manual_seed(seed)
        model = RegHead(Xtr.shape[1], LABEL_COLS).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        best, bad, best_state = -1, 0, None
        for ep in range(100):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 4096):
                idx = perm[i:i + 4096]
                x = Xtr[idx].cuda()
                loss = 0.0
                for ci, c in enumerate(LABEL_COLS):
                    z = model(x, c)
                    y = Ytr[idx, ci].cuda()
                    loss = loss + F.mse_loss(z, F.one_hot(y, N_LEVELS[c]).float())
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            mq = qwk_from_preds(Yva, preds_of(model, Xva))
            if mq > best:
                best, bad = mq, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience_default:
                    break
        model.load_state_dict(best_state)
        return model

    per = []
    for seed in (0, 1, 2):
        model = train_mse(seed)
        per.append(round(qwk_from_preds(Yte, preds_of(model, Xte)), 4))
        del model
        torch.cuda.empty_cache()
    OUT["E4_mse_regression"] = {"config": "518bm, K-way softmax-free MSE-to-onehot + argmax, 3 seeds",
                                "per_seed": per, "mean": round(float(np.mean(per)), 4)}
    print("E4 done", per, flush=True)

    e8 = {}
    from train_ordinal import train_one as _t1
    from p50b_significance import predict_kind as _pk
    _kmap = {"ce": "ce", "coral_shared": "coral_shared", "corn": "corn", "cumul": "coral"}
    for kind in ("ce", "coral_shared", "corn", "cumul"):
        per = []
        for seed in (0, 1, 2):
            model = _t1(_kmap[kind], Xtr_r, Ytr_r, Xva_r, Yva_r, epochs=100, seed=seed)[0]
            pt = _pk(model, _kmap[kind], Xte_r)
            per.append(round(qwk_from_preds(Yte_r, pt), 4))
            del model
            torch.cuda.empty_cache()
        e8[kind] = {"per_seed": per, "mean": round(float(np.mean(per)), 4),
                    "std": round(float(np.std(per, ddof=1)), 4)}
        print(f"E8 {kind} done {per}", flush=True)
    OUT["E8_loss_matrix_raddino"] = {"config": "RAD-DINO 224 CLS frozen, 3 seeds", "losses": e8}

    OUT["total_seconds"] = round(time.time() - t0)
    (PROJ / "results" / "p50_review_experiments.json").write_text(json.dumps(OUT, indent=1))
    print("P50-A ", flush=True)


if __name__ == "__main__":
    (Xtr_r, Ytr_r, _), (Xva_r, Yva_r, _), (Xte_r, Yte_r, _) = (load_split("features-raddino", "cls", s) for s in ("train", "val", "test"))
    main()
