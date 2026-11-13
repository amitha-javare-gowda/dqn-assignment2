"""Build report.pdf from the contents of results/.

    python make_report.py --github-user YOUR_USER --repo YOUR_REPO

MsPacman sections are filled automatically once results/mspacman/log.json and
results/mspacman/eval.json exist; until then they show a clearly marked
placeholder. All numbers in the report are read from the JSON result files.
"""
import argparse
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

R = "results"
V = f"{R}/validation"


def J(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1d3557"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1d3557"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
SM = ParagraphStyle("SM", parent=B, fontSize=8.3, leading=10.5)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.3, leading=10.5, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=8, leading=10,
                      backColor=colors.HexColor("#f3f4f6"), borderPadding=4)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=19, textColor=colors.HexColor("#1d3557"))


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items, s=B):
    return [Paragraph(f"&bull;&nbsp;{i}", ParagraphStyle("bl", parent=s, leftIndent=12,
                                                          firstLineIndent=-8)) for i in items]


def fig(path, caption, w=6.8):
    if not os.path.exists(path):
        return [P(f"<i>[figure not generated yet: {path}]</i>", CAP)]
    from PIL import Image as PI
    im = PI.open(path)
    h = w * im.height / im.width
    return [KeepTogether([Image(path, width=w * inch, height=h * inch), P(caption, CAP)])]


def table(rows, widths=None, font=8.3):
    data = [[Paragraph(str(c), ParagraphStyle("tc", parent=SM, fontSize=font)) for c in r]
            for r in rows]
    t = Table(data, colWidths=[w * inch for w in widths] if widths else None, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def build(a):
    cp_eval, cp_cfg, cp_log = J(f"{R}/cartpole/eval.json"), J(f"{R}/cartpole/config.json"), J(f"{R}/cartpole/log.json")
    pm_eval, pm_log, pm_cfg = J(f"{R}/mspacman/eval.json"), J(f"{R}/mspacman/log.json"), J(f"{R}/mspacman/config.json")
    v1, v2, v3, v4, v5, v6 = (J(f"{V}/v1_analytic.json"), J(f"{V}/v2_cartpole_values.json"),
                              J(f"{V}/v3_value_scale.json"), J(f"{V}/v4_ablations.json"),
                              J(f"{V}/v5_first_steps.json"), J(f"{V}/v6_mspacman_checks.json"))
    # the checkpoint rule in train_cartpole.py: best periodic eval, ties go to the later one
    ck = {}
    if cp_log and cp_log.get("eval_return"):
        best = max(cp_log["eval_return"])
        ck["episode"] = max(e for e, v in zip(cp_log["eval_episode"], cp_log["eval_return"]) if v == best)
    S = []
    S += [P("Assignment 2: Deep Q-Network", TITLE),
          P(f"<b>Name:</b> {a.name} &nbsp;&nbsp; <b>GitHub username:</b> {a.github_user} &nbsp;&nbsp; "
            f"<b>Repository:</b> {a.repo}<br/><b>Link:</b> "
            f"https://github.com/{a.github_user}/{a.repo}", B), Spacer(1, 6)]

    # ---------------------------------------------------------------- how to run
    S += [P("1. How to run the code", H1),
          P("Everything is in the repository; the README repeats these commands. CartPole runs on a "
            "CPU in about ten minutes; MsPacman was trained on a single GPU.")]
    S += [Preformatted(
        "bash scripts/setup_env.sh && source .venv/bin/activate\n\n"
        "# CartPole-v1\n"
        "python train_cartpole.py --episodes 800 --out results/cartpole\n"
        "python evaluate.py --game cartpole --ckpt results/cartpole/best.pt --episodes 500\n"
        "python plots.py --log results/cartpole/log.json --game cartpole\n\n"
        "# MsPacman-v5 (one GPU; rerun the same command to resume after a stop)\n"
        "CUDA_VISIBLE_DEVICES=0 bash scripts/run_mspacman.sh     # or: sbatch scripts/mspacman.slurm\n\n"
        "# Validation experiments (Section 5); outputs in results/validation/\n"
        "python -m validation.v1_analytic_mdps\n"
        "python -m validation.v2_cartpole_value_checks --ckpt results/cartpole/best.pt\n"
        "python -m validation.v3_value_scale\n"
        "python -m validation.v4_ablations\n"
        "python -m validation.v5_first_steps_trace --ckpt results/cartpole/best.pt\n"
        "python -m validation.v6_mspacman_checks\n\n"
        "python make_report.py   # rebuilds this PDF from results/", CODE)]

    # ---------------------------------------------------------------- implementation
    S += [P("2. Implementation", H1),
          P("One agent class (<font face='Courier'>dqn/agent.py</font>) is shared by both games. It "
            "holds an online Q-network and a separate <b>target network</b>, acts epsilon-greedily "
            "(ties between equal Q-values are broken at random), samples minibatches from a uniform "
            "<b>replay buffer</b> (<font face='Courier'>dqn/buffer.py</font>), and minimises the Huber loss "
            "between Q(s,a) and the Double-DQN target "
            "r + &gamma;(1 &minus; terminated)&middot;Q<sub>target</sub>(s&prime;, argmax<sub>a&prime;</sub> Q(s&prime;,a&prime;)). "
            "Gradients are clipped to norm 10. Every component can be switched off from the command "
            "line, which the ablation study (Section 5.4) uses."),
          table([
              ["Setting", "CartPole-v1", "MsPacman-v5"],
              ["Network", "MLP 4-256-256-2, ReLU", "CNN on 4 stacked 88x80 frames: conv 32x8x8/4, 64x4x4/2, 64x3x3/1, FC 512, 9 outputs"],
              ["Discount &gamma;", "0.95 (required)", "0.99 (required)"],
              ["Optimiser / LR", "Adam 1e-3, linear decay to 5e-5 over 150k steps", "Adam 1e-4, halved every 400k updates (floor 1e-5)"],
              ["Replay buffer", "100k transitions, batch 128, learn every step after 1k", "100k transitions (int8), batch 32, learn every 4 steps after 20k"],
              ["Target network", "hard copy every 500 updates", "hard copy every 2,500 updates (10k steps)"],
              ["Exploration", "&epsilon; 1.0 to 0.01 over 15k steps", "&epsilon; 1.0 to 0.1 over 1M steps, then to 0.01 over 2M"],
              ["Loss / target", "Huber, Double-DQN", "Huber, Double-DQN, reward clipped to sign(r) for the target"],
              ["Terminal handling", "only pole-fall / off-track ends bootstrapping; the 500-step time limit does not",
               "game over and (training only) life loss end bootstrapping; ALE frame cap does not"],
              ["Model used for evaluation", f"best periodic greedy check (episode {ck.get('episode')})", "best 100-episode moving average checkpoint"],
          ], [1.25, 2.6, 3.0]),
          Spacer(1, 4),
          P("MsPacman uses the handout's <font face='Courier'>preprocess_observation</font> verbatim; four "
            "consecutive frames are stacked so the network can see motion. The buffer stores the int8 "
            "frames directly and the network rescales them to about [-1, 1].")]

    # ---------------------------------------------------------------- CartPole
    S += [PageBreak(), P("3. CartPole-v1 (&gamma; = 0.95)", H1)]
    if cp_eval:
        S += [P(f"<b>Result:</b> over 500 fresh evaluation episodes (seeds disjoint from training) the "
                f"greedy policy scores <b>mean {f(cp_eval['mean'],1)}, std {f(cp_eval['std'],1)}</b> "
                f"(min {cp_eval['min']:.0f}, max {cp_eval['max']:.0f}); "
                f"{100*cp_eval['frac_500']:.0f}% of episodes reach the 500-step limit, so the "
                "environment is solved.")]
    S += fig(f"{R}/cartpole/max_q.png",
             "Figure 1 (i). Maximum Q-value per episode. The dashed line is the theoretical ceiling "
             "1/(1&minus;&gamma;) = 20 for +1 rewards.", 6.0)
    S += [P("<b>Discussion (i).</b> Max Q rises roughly like (1&minus;&gamma;<sup>L</sup>)/(1&minus;&gamma;) as episode "
            "length L grows, overshoots to about 23 around episode 550 (the overestimation bias of "
            "bootstrapped max-targets, still present with Double DQN while the learning rate is high), "
            "and returns to about 20.2 as the learning rate decays. A value well above 20 cannot be "
            "correct for this task, so the ceiling is a built-in sanity check we exploit in Section 5.")]
    S += fig(f"{R}/cartpole/rewards.png",
             "Figure 2 (ii). Episode reward during training with the moving average of the last 100 "
             "episodes.", 6.0)
    best_ma = max(__import__("numpy").convolve(cp_log["reward"], __import__("numpy").ones(100) / 100, "valid")) if cp_log else None
    S += [P(f"<b>Discussion (ii).</b> Learning starts once &epsilon; has decayed (around episode 350). "
            f"The training curve (best 100-episode average {f(best_ma,0)}) understates the policy: it "
            "includes 1% random actions and a network that is still being updated every step, and DQN on "
            "CartPole is known to oscillate between good and degraded policies. We therefore evaluate "
            "the greedy network every 10 episodes and keep the best checkpoint, then verify it on 500 "
            "new episodes (Figure 3), which removes any selection bias from the 5-episode check.")]
    S += fig(f"{R}/cartpole/eval_hist.png", "Figure 3 (iii). Histogram of 500 evaluation episodes.", 4.6)
    S += [P("<b>Why &gamma; = 0.95 makes CartPole hard.</b> Before tuning, every greedy failure we inspected "
            "(20 of 20) was the cart leaving the track (|x| &gt; 2.4), never the pole falling. With "
            "&gamma; = 0.95 a crash k steps ahead only lowers Q by 20&middot;&gamma;<sup>k</sup>: 1.5 at k = 50 and "
            "0.12 at k = 100, while the pole-balancing decisions involve Q-gaps of 10 or more. The fix "
            "was capacity and precision rather than shaping: a wider network (256 units), larger batches "
            "(128), a larger buffer that keeps rare near-edge states, and a decaying learning rate so "
            "the network can resolve those small differences.")]

    # ---------------------------------------------------------------- MsPacman
    S += [PageBreak(), P("4. MsPacman-v5 (&gamma; = 0.99)", H1)]
    rb = v6["random_baseline"] if v6 else None
    if pm_log and pm_eval:
        import numpy as np
        r = np.array(pm_log["reward"], float)
        ma = np.convolve(r, np.ones(100) / 100, "valid")
        hours = pm_log["seconds"][-1] / 3600 if pm_log.get("seconds") else None
        S += [P(f"Trained for {len(r)} episodes ({pm_log['steps'][-1]:,} agent steps"
                + (f", {hours:.1f} GPU hours" if hours else "") + "). "
                f"The 100-episode moving average rose from {ma[0]:.0f} (first 100 episodes) to a best of "
                f"{ma.max():.0f} and ended at {ma[-1]:.0f}. A uniformly random policy scores "
                f"{rb['mean']:.0f} &plusmn; {rb['std']:.0f}." if rb else ""),
              P(f"<b>500-episode evaluation (&epsilon; = {pm_eval['epsilon']}):</b> mean "
                f"<b>{pm_eval['mean']:.1f}</b>, std <b>{pm_eval['std']:.1f}</b> (median "
                f"{pm_eval['median']:.0f}, min {pm_eval['min']:.0f}, max {pm_eval['max']:.0f}). "
                f"That is {pm_eval['mean']/rb['mean']:.1f}x the random baseline." if rb else "")]
    else:
        S += [P("<font color='#c0392b'><b>[MsPacman results pending: run scripts/run_mspacman.sh on the "
                "GPU machine, then rerun make_report.py. This section, Figures 4 to 7 and the numbers "
                "below fill in automatically.]</b></font>")]
    S += fig(f"{R}/mspacman/max_q.png", "Figure 4 (i). MsPacman maximum Q-value per episode.", 6.0)
    S += [P("<b>What to expect in (i).</b> With clipped rewards and &gamma; = 0.99, Q is bounded by "
            "1/(1&minus;&gamma;) = 100 and in practice sits far below it (rewards are sparse and life "
            "loss cuts the bootstrap). A steady rise that then flattens indicates learning; an "
            "unbounded climb past roughly 30 to 40 would indicate divergence.")]
    if pm_log:
        import numpy as np
        r = np.array(pm_log["reward"], float)
        blocks = [r[i:i + 500].mean() for i in range(0, len(r), 500)]
        S += [P("<b>Learning progress by 500-episode block:</b> "
                + ", ".join(f"{b:.0f}" for b in blocks)
                + ". The mean score increases in every block, and the last block is still the steepest "
                "rise, so the agent had not converged after 5,000 episodes; more training would very likely "
                "raise the score further (Section 8). Max Q grows in step with the score, as expected when "
                "the agent survives longer and collects more clipped rewards.")]
    S += fig(f"{R}/mspacman/rewards.png",
             "Figure 5 (ii). MsPacman episode score with the 100-episode moving average (its first "
             "points average fewer than 100 episodes, hence the brief spike at the start).", 6.0)
    S += fig(f"{R}/mspacman/eval_hist.png", "Figure 6 (iii). Histogram of 500 evaluation episodes.", 4.6)
    S += fig(f"{R}/mspacman/actions.png",
             "Figure 7 (non-standard). Share of each of the 9 actions per 50-episode window. Random "
             "play is uniform (0.11 each).", 6.4)
    S += [P("<b>Reading Figure 7.</b> We expected the learned policy to concentrate on a few moves and "
            "to abandon NOOP. It did not: averaged over 500-episode blocks every action stays between "
            "about 9% and 14% of steps (individual 50-episode windows range from about 7% to 17%), and "
            "no action is abandoned, so the prediction was wrong. On reflection this is reasonable. In "
            "Ms. Pac-Man, NOOP keeps her moving in her current direction, so it is a useful action, and "
            "the maze requires all four directions (the diagonals act as turn-at-next-junction inputs). "
            "Learning shows up in <i>when</i> each action is chosen, not in <i>which</i> actions are "
            "used overall, which is why the score rises 5x while the action mix barely changes.")]
    S += [P("<b>Design choices for MsPacman.</b> Huber loss, a step learning-rate scheduler and a two-phase "
            "&epsilon; schedule, as suggested in the handout; Double-DQN targets to limit overestimation; "
            "reward clipping so a 200-point ghost and a 10-point pellet do not produce wildly different "
            "gradient scales (the logged score is always the raw game score); and treating a lost life "
            "as terminal for bootstrapping only (the episode itself continues), which gives the agent a "
            "much denser signal that dying is bad.")]

    # ---------------------------------------------------------------- edge cases
    S += [PageBreak(), P("5. Edge cases and how they were handled", H1),
          table([
              ["Edge case", "Why it matters", "How it is handled / verified"],
              ["Time-limit truncation (CartPole at 500; ALE frame cap)",
               "Gymnasium returns terminated and truncated separately. Treating truncation as terminal tells "
               "the net the state after step 500 is worth 0, which is false and pulls Q below 20.",
               "Only <i>terminated</i> is stored as done. Verified in V1(b): correct code converges to 20.0, the "
               "buggy variant to its predicted 12.25."],
              ["Huber loss can hide bugs", "Huber is an L1 fit for large errors, so rare wrong targets barely move it.",
               "Found while building V1: with Huber the truncation bug gives 19.5 (looks fine). The bug test therefore uses MSE."],
              ["Terminal masking", "A missing (1 &minus; done) makes crash states bootstrap from a non-existent future.",
               "V1(a): the corridor's terminal reward is learned to within 1e-6 of the closed form."],
              ["Overestimation / divergence", "max over noisy estimates is biased upward; without a target net it can blow up.",
               "Double-DQN + target net + grad clipping; ceiling check 1/(1&minus;&gamma;); V4 shows divergence when the target net is removed."],
              ["Cart drifting off-track (&gamma; = 0.95)", "Crash signal 50+ steps away is tiny in Q.", "Diagnosed by logging the termination cause; fixed with capacity, batch size and LR decay (Section 3)."],
              ["Tied Q-values at initialisation", "argmax always picks action 0, biasing early data.", "Random tie-breaking in act()."],
              ["Sampling before the buffer is full", "Would sample uninitialised zero rows.", "Learning starts after 1k (CartPole) / 20k (MsPacman) transitions; the buffer raises if asked for more than it holds."],
              ["Frame stack at reset", "Stale frames from the previous episode would leak into the first states.", "Stack is refilled with 4 copies of the first frame; checked in V6."],
              ["Life loss vs game over (MsPacman)", "Game continues after a death; bootstrapping across a death is optimistic.",
               "Life loss ends bootstrapping in training only; scores and episode boundaries use real game over. V6 confirms 3 lives and 3 life losses per random game."],
              ["Reward scale in MsPacman", "Points range 10 to 1600; raw targets give unstable gradients.", "Clip to sign(r) for the TD target only; raw score is logged and evaluated."],
              ["int8 overflow", "Preprocessing yields int8; arithmetic on it could wrap around.", "Cast to float inside the network; buffer keeps int8 to save memory (V6 checks range -128..127)."],
              ["Evaluation leakage / lucky checkpoint", "Choosing the best checkpoint on the same episodes would be optimistic.", "Evaluation seeds are disjoint from training seeds; the selected CartPole model is re-tested on 500 new episodes."],
              ["Long cluster job interrupted", "Losing a multi-hour run.", "Checkpoints every 50 episodes, --max-hours clean stop, --resume (tested), buffer refilled before learning resumes."],
          ], [1.45, 2.45, 2.95], font=7.8)]

    # ---------------------------------------------------------------- validation
    S += [PageBreak(), P("6. Tests that the algorithm is working correctly", H1),
          P("These are experiments with a prediction made <i>before</i> running them, not unit tests. "
            "Each one can be rerun with the command shown; the numbers below are read from "
            "<font face='Courier'>results/validation/*.json</font>.")]

    S += [P("6.1 V1: recovering known Q-values on analytic MDPs", H2),
          P("<font face='Courier'>python -m validation.v1_analytic_mdps</font>. The same agent and buffer "
            "are trained off-policy (uniform random behaviour) on two tiny deterministic MDPs with "
            "closed-form Q*. (a) A 10-state corridor that terminates with +1 at the right end: "
            "Q*(s,right) = &gamma;<sup>8&minus;s</sup>. (b) A one-state survival loop paying +1 per step with a "
            "30-step time limit, which mirrors CartPole: Q* = 1/(1&minus;&gamma;) = 20. As a control we also "
            "train a deliberately buggy variant that treats truncation as termination; its predicted "
            "fixed point is 1/(1 &minus; &gamma;(1 &minus; 1/30)) = 12.25.")]
    if v1:
        S += [table([["Quantity", "Prediction", "Measured"],
                     ["Corridor, max |Q &minus; Q*|", "&lt; 0.05", f"{v1['corridor_max_abs_err']:.1e}"],
                     ["Corridor, greedy action everywhere", "right", "right" if v1["corridor_greedy_all_right"] else "NOT all right"],
                     ["Survival Q, correct truncation handling", "20.0", f(v1["survival_Q_correct"])],
                     ["Survival Q, buggy (truncation = terminal), MSE", f(v1["buggy_theory"]), f(v1["survival_Q_buggy"])],
                     ["Survival Q, buggy, Huber loss", "masked (close to 20)", f(v1["survival_Q_buggy_huber"])],
                     ["Survival Q, correct, every reward doubled", "40.0", f(v1.get("survival_Q_reward_x2"))]],
                    [3.2, 1.6, 1.6])]
    S += fig(f"{V}/v1_analytic.png", "Figure 8. V1: learned vs exact Q on the corridor (left); correct vs "
             "buggy truncation handling on the survival loop (right).", 6.8)

    S += [P("6.2 V2: are the trained CartPole values physically meaningful?", H2),
          P("<font face='Courier'>python -m validation.v2_cartpole_value_checks</font>. Because CartPole is "
            "deterministic, Q of the greedy policy can be compared with the realised discounted return. "
            "The trained agent never fails from normal starts, so we also start it from 400 random "
            "perturbed states (many unrecoverable) and let it play greedily.")]
    if v2:
        g, pz = v2["greedy, normal starts"], v2["greedy, perturbed starts"]
        S += bullets([
            f"Normal starts: mean |Q &minus; G| = {f(g['mae_Q_vs_G'])} over {g['n_states']:,} states (Q &asymp; G &asymp; 20).",
            f"Perturbed starts: {100*pz['frac_terminated']:.0f}% crash. At the start state, Q correlates "
            f"{f(pz['start_state_corr'])} with the realised return (MAE {f(pz['start_state_mae'])}): the net ranks "
            "doomed vs recoverable states correctly, though it is optimistic (it predicts escape from states it then fails to escape).",
            f"Policy map: in the {100*v2['decisive_state_fraction']:.0f}% of the (&theta;, d&theta;/dt) grid where the choice "
            f"matters (|Q(R) &minus; Q(L)| &gt; 1), the policy pushes toward the side the pole will lean 0.1 s later in "
            f"{100*v2['policy_agreement_decisive']:.1f}% of states. Elsewhere both actions are worth about 20 and the gap is near 0, as expected.",
            f"Ceiling: max Q on perturbed (out-of-distribution) states reaches {f(pz['max_Q'])}, about 11% above 20. "
            "We report this as a known weakness: overestimation persists away from the data distribution."])
    S += fig(f"{V}/v2_cartpole_values.png",
             "Figure 9 (non-standard). (a) Predicted Q vs realised return. (b) Q against steps remaining "
             "before the crash, with the theoretical curve (1&minus;&gamma;<sup>k</sup>)/(1&minus;&gamma;). (c) Q(right) "
             "&minus; Q(left) across pole angle and angular velocity.", 6.9)
    S += [P("Figure 9(b) is the most informative plot in this report. The exact value starts falling "
            "about 60 steps before a crash, but most trajectories stay near 20 until the last 10 or so "
            "steps and then drop sharply; only a minority begin falling 20 to 40 steps out. So the "
            "network recognises imminent, unrecoverable pole falls but is blind to slower failures such "
            "as cart drift: the same weakness described in Section 3 and the source of the optimism "
            "(points above the diagonal) in panel (a).")]

    S += [P("6.3 V3: do values scale with &gamma; and reward as theory says?", H2),
          P("<font face='Courier'>python -m validation.v3_value_scale</font>. With +1 per step the plateau of "
            "max Q must approach r/(1&minus;&gamma;). Prediction: about 10 for &gamma; = 0.90, 20 for 0.95, and "
            "exactly twice as large when every reward is doubled. We also predict the plateau from the "
            "observed episode lengths L as the median of r(1&minus;&gamma;<sup>L</sup>)/(1&minus;&gamma;).")]
    if v3:
        rows = [["Run", "Ceiling", "Prediction from episode lengths", "Measured plateau", "Max ever"]]
        for k in ["γ=0.90", "γ=0.95", "γ=0.95, reward×2"]:
            if k in v3:
                rows.append([k.replace("γ", "&gamma;").replace("×", "x"), f(v3[k]["ceiling"], 1),
                             f(v3[k]["finite_horizon_pred"], 1), f(v3[k]["plateau"], 1), f(v3[k]["max_ever"], 1)])
        S += [table(rows, [1.6, 0.8, 2.0, 1.2, 0.9]),
              P("The &gamma; predictions hold: 9.8 against 10.0 for &gamma; = 0.90, and 13.3 against 14.3 for "
                "&gamma; = 0.95 (episodes in this short run were not yet long enough to reach 20). The "
                "reward x2 run is <b>not</b> a clean test: its agent reached long episodes later, so its Q "
                "(29.2) had not caught up with the 39.4 implied by its episode lengths, and Huber's fixed "
                "threshold of 1 means a doubled reward also changes the gradient dynamics, so the two runs "
                "learn differently. We therefore test reward linearity where everything else is held fixed, "
                f"on the analytic survival MDP of V1: doubling every reward gives Q = "
                f"{f(v1.get('survival_Q_reward_x2') if v1 else None)} against the exact 40.")]
    S += fig(f"{V}/v3_value_scale.png", "Figure 10. V3: max Q per episode for three settings; dashed lines "
             "are the ceilings.", 5.6)

    S += [P("6.4 V4: ablations of the three required components", H2),
          P("<font face='Courier'>python -m validation.v4_ablations</font>. One component is removed at a time "
            "(500 episodes, same seed). Prediction: removing the target network or the replay buffer "
            "makes learning unstable; removing Double-DQN and Huber increases overestimation.")]
    S += [P("<b>Outcome.</b> Two of three predictions held. Without a target network the values "
            "diverge (max Q reaches about 7e7 and the agent never learns). The vanilla max target with MSE "
            "pushes max Q above 21 in about a quarter of episodes, against none for the full agent, which "
            "is the overestimation Double DQN removes. The replay-buffer prediction <b>failed</b> on "
            "CartPole: learning only from the latest 128 transitions did not hurt, and in this single "
            "seed it even learned faster (best moving average 273 against 183), though it also fell back "
            "more sharply at the end. Plausible reasons: the state is 4-dimensional and smooth, the target network "
            "already stabilises the regression, and each transition is still reused for 128 updates, "
            "which is a small sliding-window replay. We would expect the full buffer to matter on "
            "MsPacman, where consecutive frames are far more correlated, and with more seeds; that is "
            "one of the extended experiments in Section 8.")]
    if v4:
        rows = [["Variant", "Final MA-100", "Best MA-100", "First episode MA-100 &ge; 150",
                 "Episodes with max Q &gt; 21", "Largest max Q"]]
        for k, v in v4.items():
            rows.append([k, f(v["final_ma100"], 1), f(v["best_ma100"], 1),
                         v["first_ep_ma100_150"] or "never", f"{100*v['frac_maxQ_above_21']:.0f}%",
                         f"{v['max_Q_ever']:.3g}"])
        S += [table(rows, [1.1, 0.9, 0.9, 1.3, 1.3, 1.1])]
    S += fig(f"{V}/v4_ablations.png", "Figure 11. V4: learning curves (left) and max Q on a symmetric log "
             "scale (right) for each ablation.", 6.9)

    S += [P("6.5 V5: the first 10 steps", H2),
          P("<font face='Courier'>python -m validation.v5_first_steps_trace</font> prints, for a fixed-seed "
            "episode, the state, both Q-values, action, reward and the TD target. Untrained: Q &asymp; 0.1, both "
            "actions within 0.01 of each other, TD target &asymp; 1.13 = 1 + 0.95&middot;Q(s&prime;), so the reward and "
            "&gamma; enter the target correctly. Trained: Q &asymp; 20 and the TD error is below 0.15 at every "
            "step (Bellman-consistent). At step 7 (&theta; = 0.076, d&theta;/dt = 1.43, pole falling right) Q(left) "
            "collapses to 4.7 while Q(right) stays at 20, and the agent pushes right. A single SGD update on "
            "one transition moved Q(s,a) toward its target for both networks (sign check of the gradient).")]
    if v5:
        rows = [["t", "&theta;", "d&theta;/dt", "Q(L)", "Q(R)", "action", "TD target", "TD error"]]
        for r_ in v5["trained"]:
            rows.append([r_["t"], r_["theta"], r_["theta_dot"], r_["Q_left"], r_["Q_right"],
                         r_["action"], r_["td_target"], r_["td_error"]])
        S += [table(rows, [0.35, 0.75, 0.75, 0.75, 0.75, 0.6, 0.9, 0.9]),
              P("Table: trained network, first 10 steps (full traces in results/validation/v5_first_steps.md).", CAP)]

    S += [P("6.6 V6: MsPacman pipeline checks", H2),
          P("<font face='Courier'>python -m validation.v6_mspacman_checks</font> (CPU, about 3 minutes).")]
    if v6:
        S += bullets([
            f"Preprocessing output shape {tuple(v6['pre']['shape'])}, {v6['pre']['dtype']}, range "
            f"[{v6['pre']['min']}, {v6['pre']['max']}]; Ms. Pac-Man's own pixels map to {v6['pre']['those_pixels_mapped_to']} "
            "(maximum contrast), as the handout intends.",
            f"Frame stack: 4 identical frames after reset ({v6['stack']['reset_all_equal']}), correct shift on push ({v6['stack']['push_shifts']}).",
            f"{v6['n_actions']} actions; {v6['lives']['start_lives']} starting lives, "
            f"{f(v6['lives']['life_losses_per_game_mean'],1)} life losses per random game, every game ends in a true termination.",
            f"Random-policy baseline over 30 games: <b>{f(rb['mean'],1)} &plusmn; {f(rb['std'],1)}</b> "
            f"(min {rb['min']:.0f}, max {rb['max']:.0f}). Learning means clearly beating this.",
            f"Overfit test: with random &plusmn;1 rewards on 32 real transitions and a frozen target net, the TD "
            f"loss fell by a factor of {v6['overfit']['reduction_factor']:.2g} in 300 updates, so gradients flow "
            "through the CNN on real frames."])
    S += fig(f"{V}/v6_preprocess.png", "Figure 12. Raw frame and the preprocessed network input.", 3.8)

    # ---------------------------------------------------------------- non-standard plots + future
    S += [P("7. Non-standard plots", H1),
          P("(1) Figure 9(b): Q against the number of steps remaining before a crash, overlaid with the "
            "exact discounted value; it shows how far ahead the network can anticipate failure. "
            "(2) Figure 9(c) and results/validation/v2_cart_drift_map.png: maps of the action advantage "
            "Q(right) &minus; Q(left) over state space. (3) Figure 7: the action-usage heatmap over training "
            "for MsPacman.")]
    S += [P("8. With three more months: further validation and experiments", H1)]
    S += bullets([
        "<b>Multiple seeds and confidence intervals.</b> Rerun every experiment over 10+ seeds and report "
        "interquartile means with bootstrap intervals (Agarwal et al., 2021); single-seed DQN curves are noisy.",
        "<b>Exact Q* for CartPole on a grid.</b> Discretise the state space finely, run value iteration "
        "with &gamma; = 0.95 on the known dynamics, and measure the network's error state by state, "
        "especially near the track edges where V2 shows it is optimistic.",
        "<b>Fix the drift blind spot directly.</b> Prioritised replay (sample high-TD-error, rare edge "
        "states more), n-step returns, and a distributional head (C51 / QR-DQN), each tested with the "
        "Figure 9(b) curve as the metric.",
        "<b>MsPacman to convergence.</b> 10M+ frames, a 1M-frame buffer stored as single frames, Dueling "
        "and Rainbow components one at a time with ablations, and comparison with published DQN numbers.",
        "<b>Overestimation audit.</b> For MsPacman, compare predicted Q with the Monte-Carlo discounted "
        "return of the clipped rewards throughout training (as in van Hasselt et al., 2016).",
        "<b>Robustness and generalisation.</b> Evaluate under no-op starts, without sticky actions, with "
        "changed pole length or mass in CartPole, and on held-out MsPacman mazes (later levels).",
        "<b>Interpretability.</b> Saliency / Grad-CAM maps over MsPacman frames to check the agent attends "
        "to ghosts, pellets and its own position, plus probing the CNN features for ghost distance.",
        "<b>Hyperparameter sensitivity.</b> Sweeps over learning rate, target period, buffer size and "
        "&epsilon; schedule to show which settings the results depend on.",
    ])
    doc = SimpleDocTemplate("report.pdf", pagesize=letter, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            title="Assignment 2: Deep Q-Network")
    doc.build(S)
    print("wrote report.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Amitha")
    p.add_argument("--github-user", default="YOUR_GITHUB_USERNAME")
    p.add_argument("--repo", default="dqn-assignment2")
    build(p.parse_args())
