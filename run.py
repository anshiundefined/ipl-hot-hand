"""
Is there a hot hand in the IPL?

Ball-by-ball data for every IPL match (Cricsheet). After a batter hits a boundary, is the next
ball more likely to go for four or six than the match situation and the batter's skill predict?
And does the bowler 'tilt' after being hit?

1. Raw conditional probabilities (what commentators see).
2. Linear probability models with batter and bowler skill, match-phase, wickets and season controls.
3. ML test: does adding streak features improve out-of-sample prediction of the next ball
   for a gradient-boosting model that already knows the context? (Train on early seasons, test on later ones.)
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from utils import DATA, HIGHLIGHT, MUTED, PALETTE, get, md_table, savefig, setup, write_results

URL = "https://cricsheet.org/downloads/ipl_json.zip"
SRC = "Cricsheet ball-by-ball IPL data (cricsheet.org)"


def load() -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(get(URL, timeout=300).content))
    rows = []
    for name in z.namelist():
        if not name.endswith(".json"):
            continue
        m = json.loads(z.read(name))
        date = m["info"]["dates"][0]
        season = int(str(date)[:4])
        for inn_no, inn in enumerate(m.get("innings", [])[:2], start=1):
            if inn.get("super_over"):
                continue
            wkts = 0
            for ov in inn.get("overs", []):
                for ball_no, dl in enumerate(ov["deliveries"]):
                    ex = dl.get("extras", {})
                    rows.append({"match": name[:-5], "season": season, "innings": inn_no, "over": ov["over"],
                                 "ball_in_over": ball_no, "batter": dl["batter"], "bowler": dl["bowler"],
                                 "runs_bat": dl["runs"]["batter"], "wide": int("wides" in ex),
                                 "wickets_before": wkts, "wicket": int(bool(dl.get("wickets")))})
                    wkts += len(dl.get("wickets", []))
    d = pd.DataFrame(rows)
    d["seq"] = np.arange(len(d))
    return d


def features(d: pd.DataFrame) -> pd.DataFrame:
    b = d[d["wide"] == 0].copy()                                   # balls the batter actually faced
    b["boundary"] = b["runs_bat"].isin([4, 6]).astype(int)
    b["dot"] = ((b["runs_bat"] == 0) & (b["wicket"] == 0)).astype(int)
    g = b.groupby(["match", "innings", "batter"], sort=False)
    b["ball_faced"] = g.cumcount()
    b["prev_boundary"] = g["boundary"].shift(1)
    b["boundaries_last3"] = g["boundary"].shift(1) + g["boundary"].shift(2) + g["boundary"].shift(3)
    b["prev_dot"] = g["dot"].shift(1)
    b["runs_so_far"] = g["runs_bat"].cumsum() - b["runs_bat"]
    gb = b.groupby(["match", "innings", "bowler"], sort=False)       # bowler's previous legal ball
    b["bowler_hit_prev"] = gb["boundary"].shift(1)
    # skill: career boundary rates *excluding the current match* (avoids mechanical correlation)
    for who in ["batter", "bowler"]:
        tot = b.groupby(who)["boundary"].agg(["sum", "count"])
        mt = b.groupby([who, "match"])["boundary"].agg(["sum", "count"])
        j = b[[who, "match"]].join(tot, on=who).join(mt, on=[who, "match"], rsuffix="_m")
        loo_n = j["count"] - j["count_m"]
        prior, k = b["boundary"].mean(), 30                            # shrink small samples to the league mean
        b[f"{who}_skill"] = ((j["sum"] - j["sum_m"]) + k * prior) / (loo_n + k)
    b["phase"] = pd.cut(b["over"], [-1, 5, 14, 19], labels=["powerplay", "middle", "death"]).astype(str)
    b.to_csv(DATA / "balls.csv.gz", index=False, compression="gzip")
    return b


def main() -> None:
    setup()
    d = load()
    b = features(d)
    md = []

    # ---- 1. Raw conditional probabilities ----------------------------------------------------------
    s = b.dropna(subset=["prev_boundary"])
    p_after = s.loc[s["prev_boundary"] == 1, "boundary"].mean()
    p_other = s.loc[s["prev_boundary"] == 0, "boundary"].mean()
    s3 = b.dropna(subset=["boundaries_last3"])
    by3 = s3.groupby("boundaries_last3")["boundary"].agg(["mean", "count"])
    sb = b.dropna(subset=["bowler_hit_prev"])
    q_after = sb.loc[sb["bowler_hit_prev"] == 1, "boundary"].mean()
    q_other = sb.loc[sb["bowler_hit_prev"] == 0, "boundary"].mean()

    # ---- 2. Regressions with controls ----------------------------------------------------------------
    reg = b.dropna(subset=["prev_boundary", "bowler_hit_prev"]).copy()
    base = "boundary ~ batter_skill + bowler_skill + C(phase) + wickets_before + np.log1p(ball_faced) + C(innings) + C(season)"
    specs = {"Raw": "boundary ~ prev_boundary",
             "+ skill & context": base + " + prev_boundary",
             "+ bowler tilt": base + " + prev_boundary + bowler_hit_prev",
             "Streak of 3": base.replace("boundary ~", "boundary ~ boundaries_last3 +")}
    res = []
    fits = {}
    for name, f in specs.items():
        data = reg if name != "Streak of 3" else b.dropna(subset=["boundaries_last3"])
        m = smf.ols(f, data=data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["match"])[0]})
        fits[name] = m
        for term in ["prev_boundary", "bowler_hit_prev", "boundaries_last3"]:
            if term in m.params:
                res.append({"Model": name, "Term": term, "Effect (pp)": 100 * m.params[term],
                            "95% CI low": 100 * m.conf_int().loc[term, 0], "95% CI high": 100 * m.conf_int().loc[term, 1],
                            "p": m.pvalues[term], "N": int(m.nobs)})
    res = pd.DataFrame(res)

    # ---- 3. ML: does streak information help predict the next ball? -----------------------------------
    ctx = ["batter_skill", "bowler_skill", "over", "ball_in_over", "wickets_before", "ball_faced", "runs_so_far", "innings"]
    streak = ["prev_boundary", "boundaries_last3", "prev_dot", "bowler_hit_prev"]
    ml = b.copy()
    seasons = sorted(ml["season"].unique())
    cut = seasons[int(len(seasons) * 0.7)]
    tr, te = ml[ml["season"] < cut], ml[ml["season"] >= cut]
    out = []
    for name, feats in [("Context only", ctx), ("Context + streaks", ctx + streak)]:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)
        clf.fit(tr[feats], tr["boundary"])
        p = clf.predict_proba(te[feats])[:, 1]
        out.append({"Model": name, "Test log loss": log_loss(te["boundary"], p), "Test AUC": roc_auc_score(te["boundary"], p)})
    ml_tab = pd.DataFrame(out)
    ml_tab["Improvement vs context (log loss, %)"] = 100 * (1 - ml_tab["Test log loss"] / ml_tab["Test log loss"].iloc[0])

    # ---- Figures ------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.bar(by3.index.astype(int).astype(str), 100 * by3["mean"], color=[MUTED, PALETTE[0], PALETTE[1], HIGHLIGHT][:len(by3)])
    for i, (mv, n) in enumerate(zip(by3["mean"], by3["count"])):
        ax.text(i, 100 * mv + 0.3, f"{100 * mv:.1f}%\n(n={n:,})", ha="center", fontsize=8.5)
    ax.set_xlabel("Boundaries in the batter's previous 3 balls")
    ax.set_ylabel("P(boundary on next ball), %")
    ax.set_title("Raw conditional probabilities")
    ax = axes[1]
    r2 = res[res["Term"] == "prev_boundary"]
    y = np.arange(len(r2))[::-1]
    ax.errorbar(r2["Effect (pp)"], y, xerr=[r2["Effect (pp)"] - r2["95% CI low"], r2["95% CI high"] - r2["Effect (pp)"]],
                fmt="o", color=PALETTE[0], capsize=4, ms=7)
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y, r2["Model"])
    ax.set_xlabel("Effect of a boundary on the previous ball (pp)")
    ax.set_title("After controlling for skill and match situation")
    f1 = savefig(fig, "01_hot_hand", SRC)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(ml_tab["Model"], ml_tab["Test log loss"], color=[MUTED, HIGHLIGHT])
    lo = ml_tab["Test log loss"].min()
    ax.set_xlim(lo * 0.99, ml_tab["Test log loss"].max() * 1.003)
    for i, v in enumerate(ml_tab["Test log loss"]):
        ax.text(v, i, f" {v:.4f}", va="center")
    ax.set_xlabel(f"Out-of-sample log loss, seasons {cut}+ (lower = better)")
    ax.set_title("Does knowing the streak help a model predict the next ball?")
    f2 = savefig(fig, "02_ml_test", SRC)

    phase = s.groupby(["phase", "prev_boundary"])["boundary"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(phase))
    ax.bar(x - 0.18, phase[0.0], width=0.36, color=MUTED, label="Previous ball not a boundary")
    ax.bar(x + 0.18, phase[1.0], width=0.36, color=HIGHLIGHT, label="Previous ball a boundary")
    ax.set_xticks(x, phase.index)
    ax.set_ylabel("P(boundary), %")
    ax.set_title("Part of the raw 'hot hand' is just the phase of the innings")
    ax.legend()
    f3 = savefig(fig, "03_by_phase", SRC)

    # ---- Write-up -------------------------------------------------------------------------------------------
    ctrl = res[(res["Model"] == "+ bowler tilt")].set_index("Term")
    md.append("### Headline numbers\n")
    md.append(f"- {b['match'].nunique():,} matches, {len(b):,} legal balls, seasons {b['season'].min()}–{b['season'].max()}.")
    md.append(f"- Raw: P(boundary) after a boundary **{100 * p_after:.1f}%** vs **{100 * p_other:.1f}%** otherwise "
              f"(a {100 * (p_after - p_other):+.1f} pp 'hot hand').")
    pb = ctrl.loc["prev_boundary"]
    md.append(f"- With batter/bowler skill, phase, wickets, balls faced, innings and season controls: "
              f"**{pb['Effect (pp)']:+.2f} pp** (95% CI {pb['95% CI low']:+.2f} to {pb['95% CI high']:+.2f}).")
    bt = ctrl.loc["bowler_hit_prev"]
    md.append(f"- Bowler 'tilt': after the bowler was hit for a boundary on his previous ball, the next ball goes for a boundary "
              f"**{bt['Effect (pp)']:+.2f} pp** more often (raw gap {100 * (q_after - q_other):+.1f} pp).")
    md.append(f"- ML test: adding streak features changes out-of-sample log loss by "
              f"**{ml_tab['Improvement vs context (log loss, %)'].iloc[1]:+.2f}%** (AUC {ml_tab['Test AUC'].iloc[0]:.3f} → {ml_tab['Test AUC'].iloc[1]:.3f}).\n")
    md.append("### Regression estimates (linear probability, SEs clustered by match)\n")
    md.append(md_table(res, "{:.3f}"))
    md.append("\n### ML: train on early seasons, test on later ones\n")
    md.append(md_table(ml_tab, "{:.4f}"))
    md.append("\n### Figures\n")
    for f, cap in [(f1, "Hot hand: raw vs controlled"), (f3, "By phase of innings"), (f2, "ML test")]:
        md.append(f"**{cap}**\n\n![{cap}]({f})\n")
    write_results("\n".join(md))
    print("done")


if __name__ == "__main__":
    main()
