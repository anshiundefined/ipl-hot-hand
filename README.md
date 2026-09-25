# Is There a Hot Hand in the IPL?

"He's seeing it like a football now." Commentators, captains and fans believe in momentum: once a batter hits a four, the next boundary is more likely. Behavioural economics has argued about the **hot hand** for 40 years. Gilovich, Vallone & Tversky (1985) called it a cognitive illusion in basketball. Miller & Sanjurjo (2018) showed the original test was biased and the hot hand may be real after all.

Cricket suits the question. Every ball is recorded, and the match situation (overs left, wickets in hand, who is bowling) can be controlled for precisely. This repo tests:
1. **Batter hot hand.** Does a boundary make the next ball more likely to go for four or six?
2. **Bowler tilt.** After being hit, does a bowler concede more?
3. **The ML test.** If streaks carry real information, a model that already knows the match context should predict the next ball *better* when it is also given the streak.

## Data
[Cricsheet](https://cricsheet.org) ball-by-ball JSON for **every IPL match since 2008**, downloaded on each run. Wides are dropped because the batter did not face them. Super overs are excluded.

## Method
**Controls that matter.** Raw streaks mislead because boundaries cluster for boring reasons: the powerplay and death overs, set batters, weak bowlers. Every model conditions on:
- **Batter skill and bowler skill.** Career boundary rates computed *leaving out the current match* (so a hot day doesn't inflate its own skill measure), shrunk towards the league mean.
- Phase of the innings, wickets fallen, balls faced (how "set" the batter is), innings, season.

**Estimators**
- Linear probability models with SEs clustered by match. Three streak measures: previous ball a boundary, boundaries in the last 3 balls, and the bowler hit on his previous ball.
- **Gradient boosting (histogram GBM)** trained on the first ~70% of seasons and tested on the rest. It is run twice, with context features only and with context plus streak features. If the hot hand is real and exploitable, out-of-sample log loss should fall.

## Results
<!-- RESULTS:START -->
_Results, tables and figures are generated automatically by the GitHub Action (`Actions` tab → **Run analysis**). They appear here a few minutes after the first push._
<!-- RESULTS:END -->

## Reproduce
```bash
pip install -r requirements.txt
python run.py
```
The GitHub Action downloads the latest Cricsheet archive and re-runs on push and monthly, so each new IPL season is added automatically.

## Limitations
- Streaks are measured within a batter's innings. Miller & Sanjurjo's small-sample bias applies to within-sequence *averages*, not to the pooled regressions used here, but short innings still limit power for the 3-ball streak.
- Field placements change after a boundary. A captain who reacts to a streak can hide a real hot hand, which is a genuine identification problem.
- Skill is measured as career averages. Form over recent matches is a separate question from momentum within an innings.

## References
- Gilovich, T., Vallone, R. & Tversky, A. (1985). *The hot hand in basketball: on the misperception of random sequences.* Cognitive Psychology.
- Miller, J. & Sanjurjo, A. (2018). *Surprised by the hot hand fallacy? A truth in the law of small numbers.* Econometrica.
- Bocskocsky, A., Ezekowitz, J. & Stein, C. (2014). *The hot hand: a new approach to an old "fallacy".* MIT Sloan Sports Analytics Conference.
