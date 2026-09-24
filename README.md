# jev-cascade

A small, readable agent loop where a cheap model does the cheap work, a strong
model is paid for only when it earns it, and [TypeSafe System One (Jev)](https://docs.typesafe.ai/llms.txt)
does the deciding: which tier runs each step, whether the result is good enough,
and, for a whole class of steps, the work itself.

The measurement section below reports how well that actually works, and the honest
answer is "partly": the cheap tier handles most steps, but the verification gate barely
separates good output from bad, so in practice the loop's value is mostly *do not use the
strong model by default*, not *detect the hard steps*.

No third-party dependencies. Python 3.11 or newer (uses `tomllib`). No keys are
needed to read the code, run the tests, or watch a full run in `--dry-run` mode.

## The idea in one paragraph

Most agent loops send every step to one model. Either everything goes to a strong
model (and you pay strong-model prices for renaming a variable), or everything
goes to a cheap model (and the hard 20% of steps quietly come back wrong). This
loop splits a task into steps, then makes two separate decisions per step: **who
runs it**, and **did it work**. Both are cheap questions with a small set of
acceptable answers, which is exactly the shape System One is built for. Jev is
not an LLM and cannot chat or write code, so it is never asked to generate
anything. It is asked to choose, to score, and to judge, which it does for a
fraction of a cent.

## What Jev actually does here

| Role | Question type | Where |
|---|---|---|
| Step typing (with the planner) | planner output shape | steps whose answer is a fixed choice, a yes/no, or an ordered position are marked `decide` |
| **Routing** | `choice` over the configured tiers, plus `jev` | one call per step, picks the executor |
| **Execution** | `choice`, `noul`, or `score` | `decide` steps are answered by System One directly, with zero model tokens |
| **Candidate selection** | `choice` over N cheap drafts | optional, replaces "pay a strong model to pick" |
| **Verification** | `noul` (probability that the step's criteria hold) | gates every generative step, and decides whether to escalate |

The fifth one is the money saver: a cheap draft that passes the gate never touches
the strong model, and a draft that fails is re-run exactly once, on the tier above.

## The policy ladder, cheapest move first

When a step's result comes back, the loop climbs a ladder rather than jumping
straight to the big model:

1. **Accept** it if the gate passes (`escalate_below`).
2. **Retry the same tier** (`same_tier_retries`) before paying for anything bigger.
   Same model, fresh sample, and it keeps the best attempt the gate saw.
3. **Retry on a transient failure** (`error_retries`, default 1). A rate limit, a
   dropped connection, or a reasoning model that burns its entire output cap
   thinking are all transient. Retrying the same tier is far cheaper than escalating,
   and escalation is what happens if the retry also fails.
4. **Escalate one tier up** (`max_escalations`), then keep whichever result the gate
   liked better, even if that is the cheap one.
5. **Stop escalating** once the run crosses `budget_guard_ratio` of its budget, so a
   long task degrades into "cheap and possibly imperfect" instead of dying halfway
   through with money spent and steps unfinished.

Every one of those decisions is recorded per step, so a run can be audited afterwards
rather than taken on faith.

## Measure the threshold instead of guessing

`escalate_below` is the single number that decides whether this loop saves money or
quietly ships bad work. Guessing it is how a cascade becomes theatre, so there is an
eval that measures it against real models:

```bash
python3 -m jev_cascade eval --pairs 32 --out report.json      # judged by the strongest tier
python3 -m jev_cascade eval --judge judge --out outside.json  # judged by an outside family
```

`--judge TIER` and `--strong TIER` override which tier grades and which tier is treated as
the strong side, so an A/B between two judges is two commands against one config.

For every generative step in `evals/tasks.toml` it builds a **pair**: the same prompt
sent to the cheap tier and to the strong tier. Then it asks four questions about that
pair:

| Question | How |
|---|---|
| Is the cheap output good enough? | Jev `Noul` against the step's own criteria |
| Where does the strong output land? | the same question, for a reference point |
| Does the gate have teeth? | a deliberately truncated copy of the cheap output is scored too |
| Is the cheap output *actually* enough? | a **blind judge** rules on both, in randomized slot order |

The judge never learns which output came from which tier, and because the order is
randomized per pair, position bias is measured rather than assumed. With those labels
the sweep is arithmetic: for each candidate threshold it reports how often the gate
would escalate, how much of that escalation was **wasted** on output that was already
good enough, how often a bad output would have been **kept**, how often an obviously
truncated output would have been **waved through**, and the resulting **cost per step**.
The recommendation is the cheapest threshold whose miss rate stays inside
`[eval] miss_tolerance`, and the whole curve is printed so the trade-off is visible.

Three deliberate refusals to look smarter than the data:

- **Too few judged pairs** and it recommends nothing, it reports the number it has.
- **A threshold that escalates every step** is never recommended, because that is not a
  cascade, it is paying for the strong model with extra steps.
- **Failed generations and failed judgements are counted and excluded**, never
  silently dropped, so a half-broken run cannot masquerade as a clean one.

`[eval] judge_tier` defaults to the strongest tier, which means the strong model is
grading a pair that contains its own output. A model grading itself is biased toward
itself, and that bias pushes the recommendation toward escalating too eagerly.

Point it at a judge instead:

```toml
[[tiers]]
name        = "judge"
kind        = "openai"          # or "anthropic" for a Claude-shaped proxy
base_url    = "https://gpt-agent.cc/v1"
model       = "gemini-3.1-pro"
api_key_env = "GPT_AGENT_KEY"
judge_only  = true
```

`judge_only` is the load-bearing word. Without it, a tier declared last is *the strong
side of every pair* and the top of the escalation ladder, so the judge would both author
and grade the same output. A `judge_only` tier is excluded from Jev's routing options,
from the escalation ladder, from `--tier`, from `[agent] planner_tier`, and from the
`strong_tier` slot; the eval reports which two tiers it compared and which one graded,
and says in the report whether the judge was the strong side (self-grading) or an
outside tier.

### What it measured here

Seven live runs against real System One, on two different model platforms, with four
different judges. Every report is kept in `evals/results/` so these numbers can be audited
rather than taken on trust. Runs A and B were the first small probe, C2 and D the first real
sample on DeepSeek's own tiers, and G and H the same measurement re-run on a cheaper
multi-model gateway."

**Runs A and B (2026-09-20)** were the first measurement, 12 pairs each on a 6-task set:
roughly 10 cents a run, and small enough that the report kept saying so.

| | run A | run B |
|---|---|---|
| judged pairs | 11 of 12 | 12 of 12 |
| judge verdicts | 9 cheap-ok, 2 strong-better | 11 cheap-ok, 1 strong-better |
| gate on real vs truncated outputs | 0.90 vs 0.45 | 0.90 vs 0.39 |
| truncated outputs waved through at 0.60 | **36%** | **17%** |
| gate discrimination (sufficient vs not) | **-0.01** | **+0.11** |
| routing: under / over | 2 / 3 of 11 | 0 / 2 of 12 |

Run A used the original verification question ("is this complete, correct and usable?")
and came back with a gate that said 0.90 about the sufficient cheap outputs and 0.92 about
the insufficient ones: no discrimination at all, which means the threshold was decoration.
Run B rephrased the same question adversarially ("check every requirement; answer no if
anything is missing, vague, or left as an exercise") and the gap turned positive, but it
rested on a single insufficient sample, so it was a direction rather than a proof. The task
set was too small and too easy, which is what the next session fixed.

**Runs C2 and D (2026-09-21)** are the real sample: a 16-task set, difficulty interleaved,
32 pairs budgeted, and the same config judged twice, once by the strong tier grading
itself and once by an outside family (`gemini-3.1-pro`, a `judge_only` tier).

| | run C2 (self-judging) | run D (outside judge) |
|---|---|---|
| pairs, judged | 32, **28 judged** | 28, **26 judged** |
| tasks sampled | all 16 | all 16 |
| cost | $0.77 | $0.56 |
| judge verdicts | 25 cheap-ok, **3 strong-better** | 21 cheap-ok, **5 strong-better** |
| gate on real vs truncated outputs | 0.79 vs 0.36 | 0.84 vs 0.34 |
| gate discrimination | **+0.11** | **-0.03** |
| routing: under / over | 1 / 13 of 28 | 1 / 6 of 26 |
| recommended threshold | 0.90 (missed 4%) | 0.90 (least-bad, missed 8%) |

Read against the sweeps, on the thresholds that matter:

| threshold | escalates | wasted | missed | corrupt pass | cost/step |
|---|---|---|---|---|---|
| 0.70 | 11% / 4% | 7% / 4% | 7% / **19%** | 14% / 15% | 0.00356 / 0.00209 |
| 0.80 | 21% / 15% | 18% / 15% | 7% / 19% | 4% / 8% | 0.00492 / 0.00321 |
| 0.90 | 57% / 62% | 50% / 50% | 4% / 8% | 4% / 0% | 0.00946 / 0.00766 |

(cost/step is self-judge / outside judge; all-strong for comparison is 0.01406 and 0.01067.)

What these two runs actually say:

- **The cheap tier is sufficient for most steps, on a task set built to be hard.** 25 of 28
  judged steps for the self-judge, 21 of 26 for the outside judge. Half the set was written
  with "strong should win" as its intent, and it mostly did not. The value of the split here
  is "do not use the strong model by default", not "detect the hard steps".
- **The gate's discrimination is the weak link, and the two judges disagree about it.**
  +0.11 when the strong tier grades itself, -0.03 when an outside family does. A negative gap
  means the gate scored the outputs the judge called *insufficient* marginally higher, so at
  that signal the threshold cannot be a detector. It is a cost dial: escalation picks a
  roughly arbitrary slice of steps, and a higher threshold catches more shortfalls by
  covering more of them, not by aiming at them.
- **What the gate does reliably catch is corruption.** Truncated outputs score 0.34 to 0.36
  against 0.79 to 0.84 for real ones, in both runs. That is the one signal worth banking.
- **Threshold 0.90 is what both runs recommend, and it is expensive.** It escalates 57 to 62%
  of steps with half of that escalation wasted on output that was already good enough, and it
  saves only 16 to 20% against sending everything to the strong tier. It was applied to
  `config.toml` because both runs recommended it and the miss rate is the stated criterion,
  with the cost consequence recorded next to the setting rather than discovered later.
- **Judge choice moves about a quarter of the verdicts.** The two runs sampled the same 16
  tasks, but the planner decomposes differently each run, so only 20 pairs overlap. On the 17
  pairs both judges ruled on, they agreed 13 times (76%): the outside judge was harsher on the
  cheap output in 3 cases and more generous in 1. A recommendation that is stable across both
  judges is worth more than one from either alone, which is why the A/B was run.

**Runs G and H (2026-09-21, on a cheaper multi-model gateway)** exist because the split is
only worth having if the *cheap* tier is genuinely cheaper. This box was pointed at
`api.llm-token.cn` instead: `step-3.7-flash` as cheap (CNY 0.4/2), `qwen3.7-plus` as strong
(CNY 1.6/8), and two outside judges from yet other families, `mimo-v2.5-pro` (CNY 1.6/8) and
`doubao-seed-2.1-turbo` (CNY 1.2/6). 16 pairs each, both judged, both all 16 tasks budgeted.

| | run G | run H |
|---|---|---|
| judge | `mimo-v2.5-pro` | `doubao-seed-2.1-turbo` |
| judged pairs | 16 of 16, **no failures** | 14 of 16, 1 generation and 1 judge error |
| cost | CNY 0.54 | CNY 0.44 |
| judge verdicts | 12 cheap-ok, **4 strong-better** | 12 cheap-ok, **2 strong-better** |
| gate discrimination | **+0.09** | **+0.07** |
| routing: under / over | 2 / 6 of 16 | 0 / 5 of 14 |
| cheapest threshold inside tolerance | 0.90 (missed 0%) | 0.90 (missed 0%) |

What these two add to the picture:

- **A cheaper cheap tier makes the split look worse, not better.** The insufficiency rate
  rose from 19% (measured on `deepseek-flash` in run D) to **25%** on `step-3.7-flash`. The
  cheaper the cheap tier, the more often the gate escalates, which is exactly the tension
  the eval exists to expose.
- **At 0.90 the cascade stops paying for itself on these models.** Run G: a step costs
  CNY 0.01244 against CNY 0.01258 for sending everything to the strong tier, a **0% saving**.
  The 4.2x price ratio between the tiers is real, but escalating 75% of steps spends it back.
  A threshold this high is only defensible if you value the 11% of shortfalls it catches; as a
  cost strategy it is not one.
- **Both outside judges still recommend 0.90**, and they agree less with each other than the
  previous pair did: on the 10 pairs both ruled on, they agreed 6 times (60%), against 76% for
  the DeepSeek pair. Three of the four disagreements were `mimo-v2.5-pro` being the harsher
  judge. Judge identity is a bigger variable than the threshold.
- **Both runs report a positive discrimination gap** (+0.09 and +0.07), unlike run D's -0.03,
  so the gate is not decoration on these models. It is still a weak signal: 0.85 versus 0.76
  on 12 sufficient against 4 insufficient outputs is not much separation to hang 75% of your
  escalations on.

Two things learned about the platform itself, which apply to any gateway you point tiers at:

- **Not every model honours `response_format = {"type": "json_object"}`,** and the planner and
  judge both depend on it. Measured on this endpoint: `step-3.7-flash`, `kimi-k3`, `kimi-k2.7`,
  `deepseek-v4-pro`, `deepseek-v4.1-flash`, `doubao-seed-2.1-turbo`, `mimo-v2.5-pro` and
  `grok-4.6` return real JSON; `glm-5.3`, `glm-5.3-flash`, `qwen3.7-plus` and `MiniMax-M3`
  answer in prose, and every pair they graded came back unparsed. A model that ignores the
  flag is still fine as an *executor*, since generation needs no JSON. This is why the judge
  tiers in `config.toml` are not the strongest models available.
- **The default 120s timeout is too short here, and a large `max_tokens` is actively harmful.**
  Two of 32 pairs were lost to read timeouts before `timeout_s` was raised to 300. Separately,
  carrying over `max_tokens = 16384` from the DeepSeek reasoning tiers produced single
  generations that took **4 to 7 minutes**, because these models actually use the cap; at 4096
  the same work runs at 20 to 30 seconds. A cap is not free merely because it is unused when
  the model that uses it is not a thinking model.

Five configuration defects this measurement exposed, every one of them found by running the
thing rather than by reasoning about it, and every one fixed rather than worked around. The first
uncapped run of the day (kept as `evals/results/20260921-run-c-selfjudge-24pairs-uncapped.json`)
is the one that found the first three, which is why it is filed next to the clean runs rather
than deleted:

- **The planner's own cap was too small.** At `planner_max_tokens = 2048`, 5 of 16 tasks
  returned truncated JSON and fell back to a generic single step. At 4096, none did.
- **The judge's cap was too small for a reasoning judge.** At `judge_max_tokens = 4096`, 3 of
  24 judge calls came back empty (`finish_reason=length`). Raised to 8192.
- **The strong tier burns its cap exactly on the hardest steps.** 6144 dropped a hard step,
  then 12288 dropped the zero-downtime migration step the same way. Raised to 16384 on those
  models, where thinking consumes the cap.
- **A dropped connection used to end the whole run.** A `RemoteDisconnected` is a
  `ConnectionError`, not a `URLError`, so it escaped the Jev retry loop and killed an entire
  16-pair eval mid-flight. Connection-level failures now retry on the same budget as a 529 and
  surface as a typed error, and a failed gate or routing call drops one pair instead of the run.
- **A large `max_tokens` carried over from a reasoning tier is a latency bug.** On the gateway
  models, 16384 produced single generations of 4 to 7 minutes while 4096 does the same work in
  20 to 30 seconds. The setting was correct for a thinking model and wrong for everything after
  it, which is the kind of thing a config comment does not catch but a stopwatch does.

## Two decision engines behind one protocol

`[jev] kind` chooses which engine answers the typed questions:

| | `kind = "http"` (hosted) | `kind = "laya"` (local, MLX) |
|---|---|---|
| runs on | someone else's server | this Mac, in process |
| per call | about $0.0002 | nothing |
| latency, measured here | 0.13 to 0.38 s | 0.03 to 0.07 s |
| screen state leaves the machine | yes | no |
| context it can take | 32k | 512 by default, 1024 on some checkpoints |
| generates text | no | no |

Both answer the same three typed questions, so the loop cannot tell them apart, and the
choosing is the only difference. Which one to point at which role is a measured question,
not a taste one. On 8 to 16 decisions per condition, with a positive control of 7/8 to
prove the harness itself was not the problem:

| role | local Laya | hosted Jev |
|---|---|---|
| routing a request to a category | 7/8 | not run |
| a yes/no triage question | 7/8 | not run |
| reading facts off a page (extraction) | 4/4 | not run |
| picking the right control to press | 3/3 | 3/3 |
| **choosing the next action** | **1/6** | **5/6** |

So: **the local engine covers the classification-shaped calls** (the tier router, the
verification gate, `decide` steps), and **the action decision stays on the hosted engine**.
`[browser] actor` and `[computer] actor` encode that, and `browse` and `computer` both use
the hosted engine even when `[jev] kind` is `laya`.

Two limits are worth knowing before pointing anything at the local engine. Its context is
512 tokens on the English checkpoint and that budget holds the instructions, the option
labels and the state together, so it refuses a question with more than about 126 options
and silently truncates a long state, which this port clips and counts instead. Raising
`max_len` and `head_max_len` is the one measured way around the option ceiling (255 options
at 2048/1024, 111 ms) and it is outside the project's published validation, which is why it
is opt-in and off by default.

```toml
[jev]
kind = "laya"           # local, free, for routing, gates and decide steps
model = "aac6fef/laya-mlx"
# max_len = 2048        # optional, both together, raises the option ceiling
# head_max_len = 1024
```

## Browser use, the cheap way

`browse` drives a real browser toward a goal. It is the same trick as the rest of this
project, applied to a screen instead of a task:

1. **Perceive deterministically.** The page reports its own text and its own controls, with
   roles and labels. A browser can already tell you that about itself, so **no vision model
   is involved** and nothing is guessed from pixels. That is where the cost saving is.
2. **Decide with one typed question.** A single Choice over mutually exclusive options, each
   naming both the action and its target, so choosing is a closed question.

A live run against a real website, the whole thing:

```bash
python3 -m jev_cascade browse "find the download page for Python for macOS" \
    --url https://www.python.org/ --act
```

```
  step 1: click the link 'Downloads' (conf=1.00)     clicked "Downloads"
  step 2: click the link 'macOS' (conf=0.96)         clicked "macOS"
  step 3: stop because the goal is already achieved (conf=0.56)

decisions 3  engine 0.00078 USD
```

Three decisions, under a tenth of a cent, and it found the page. Compare that with the
screenshot-to-a-frontier-model loop the same job usually takes.

The option set is closed and generative-free: click any named control, scroll either way,
go back, press escape to dismiss an overlay, wait, or stop. What a page cannot do is say
what *text* belongs in it, and neither decision engine generates text.

### The writer: where typed words come from

Jev cannot generate, but the cascade already carries tiers that can, so a step that needs
a string (a search query, a short message) has two honest sources:

- **`--text`** supplies one prepared string, as before. Nothing is generated.
- **`writer_tier`** names a tier that composes the string per field: the cheapest tier
  writes it, a Jev `Noul` gate checks the draft against the goal *before anything is
  typed*, one retry, then the step refuses rather than type an unverified string. The
  writer's tokens and cost are booked in the ledger under its own tier, exactly like any
  other spend. Fields are then offered once each, as `type into the {role} '{label}'` —
  never twice, as a click and a field, because that near-duplicate is how engines stall.

```toml
[browser]
writer_tier = "cheap"   # cheap model writes, Jev decides, ledger records
```

Nothing is clicked or typed unless `--act` is passed; without it, one step is reported and
the browser is left alone. Two consecutive steps with no change on the page stop the loop,
as does a confidence below `[browser] min_confidence`, so a confused engine stalls instead
of clicking wildly. Each step is written to the ledger with its decision, its confidence and
what it cost.

The browser itself lives behind a small Node bridge (`tools/browser_bridge.mjs`) so that this
package keeps its zero dependencies: one long-lived process, one command per line, so the
page keeps its state between steps. Node and Playwright are needed only for `browse`, and the
tests for it skip themselves when they are not installed.

## Computer use, the same trick on the desktop

`computer` points the identical loop at macOS. The expensive way to drive a desktop is to
ship screenshots to a big model every step; the cheap way is to let the app say what is on
it. The frontmost app reports its own controls — buttons, checkboxes, text fields, menu
items, complete with menu paths like `File > New Window` — through the **Accessibility
tree**, read by a small Swift bridge (`tools/macos_bridge.swift`). No pixels are read, no
vision model is involved, and **Screen Recording is never requested**: the one permission
the tool needs is Accessibility, and `--check-permissions` asks macOS to show its dialog
and prints exactly what to click.

```bash
python3 -m jev_cascade computer --check-permissions        # grant, once, for your terminal
python3 -m jev_cascade computer "open a note that says hello" --app Notes --act
```

The loop is the browser's, step for step: perceive (app, window, visible text, controls,
running apps), decide (one Choice, every option naming its action and target: `click the
button 'New Tab'`, `set the text field 'Search'`, `switch to the app 'Notes'`, `press
cmd+w` via the key option, scroll, escape, wait, stop), execute, record. Text fields are
set by a direct AXValue write where the app allows it — deterministic, no keystroke
injection — falling back to focus-and-type where it does not. A click uses AXPress, with a
real mouse event at the control's own position as the fallback, never a guessed coordinate.
The writer works here too (`[computer] writer_tier`), gated by Jev exactly as in browse.

Two guards a desktop needs that a page does not:

- **Nothing acts without `--act`.** A dry run reads the screen and reports one step.
- **`[computer] allowed_apps`** names the apps the loop may act on (matched against the
  frontmost app's name or bundle id). When it is set and the frontmost app is not on it,
  the run ends before asking the engine anything. Acting on a desktop is less reversible
  than clicking a web page; the config says so.

The bridge is compiled once to a cached binary next to the source (`.gitignored`) and run
from that, or by the `swift` interpreter when compilation is unavailable. Swift is needed
only for `computer`; the rest of the cascade does not need it, and the desktop tests skip
themselves when the toolchain or the permission is missing.

## Quickstart

```bash
cd Coding/jev-cascade

make demo      # full run, stub tiers and stub Jev, no keys, no network
make test      # 279 offline checks: routing, gates, retries, budget, ledger, the eval,
               # the browser loop, the desktop loop, the writer
make check     # validate your config and see which keys are present
```

Then make it real:

```bash
cp config.example.toml config.toml     # edit the tiers
export TYPESAFE_API_KEY=...            # from console.typesafe.ai/settings/keys
export DEEPSEEK_API_KEY=...            # or whichever provider your tiers name

python3 -m jev_cascade check
python3 -m jev_cascade run "Classify the severity of this bug report, and then write the one paragraph reply we send back to the reporter."
```

## How a run flows

```
task
 |
 |-- planner (cheap tier, or the offline heuristic planner)
 |     decompose into steps, type each one: generate | decide
 |
 |-- for each step:
 |      |
 |      |-- budget and timeout check, otherwise the step is recorded as skipped
 |      |
 |      |-- Jev Choice: which executor?
 |      |     . cheap tier        a small model
 |      |     . expensive tier    a strong model
 |      |     . jev               System One answers it (decide steps only)
 |      |
 |      |-- execute
 |      |     . decide step, routed to jev -> Choice / Noul / Score, no model tokens
 |      |     . generate step -> the tier writes it
 |      |           and if candidates > 1, Jev picks the best draft
 |      |
 |      |-- execute, retrying the same tier on a transient provider error
 |      |
 |      |-- Jev Noul: does this output satisfy the step's own criteria?
 |      |     p >= escalate_below  -> keep it
 |      |     p <  escalate_below  -> retry the same tier (same_tier_retries), keeping
 |      |                            the best attempt the gate saw, and only then
 |      |                            escalate one tier up (max_escalations)
 |      |     inside the budget guard band -> stop escalating and keep the best attempt
 |      |
 |      |-- ledger: tier, model, tokens, cost, Jev calls, verdict,
 |                  retries, error retries, escalations
 |
 |-- summary: what was spent, and what the same tokens would have cost elsewhere
```

## Configuration

Everything provider-related lives in TOML, and secrets live in the environment
only. Any OpenAI-compatible endpoint works, so DeepSeek, OpenRouter, Groq,
Together, vLLM, Ollama and LM Studio are all just config edits.

```toml
[[tiers]]                       # declared cheapest first: escalation walks up this list
name        = "cheap"
kind        = "openai"
base_url    = "https://api.deepseek.com/v1"
model       = "deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"
description = "a small fast model, fine for mechanical edits, short rewrites and routine code"
price       = { input = 0.27, output = 1.10, currency = "USD" }
```

Two fields are worth care:

- **`description`** is the only thing Jev sees when it picks an executor. Write it
  as a truthful summary of what that tier is good for. A tier described as
  "a strong model" will win routing for reasons that have nothing to do with cost.
- **`price`** drives the ledger. Use your real list prices; wrong prices make the
  cost report wrong, and a wrong cost report is worse than no cost report. The currency is
  read once, from the first tier, so keep every tier's `currency` the same: mixing USD and
  CNY in one file adds up numbers that are not the same unit, and the reported total would
  then be silently wrong rather than obviously wrong.

The `[jev]` block controls the decision policy:

```toml
[jev]
model          = "jev-latest"
api_key_env    = "TYPESAFE_API_KEY"
verify         = true      # gate every generative step
escalate_below = 0.90      # Noul probability under which the step is escalated; measured, see above
max_escalations = 1        # per step
candidates     = 1         # cheap drafts for Jev to choose from (try 3)
max_state_chars = 12000    # state is clipped before it is sent, which bounds Jev cost
```

## The ledger is deliberately honest

`--ledger run.jsonl` appends one JSON object per event, and every run prints a
summary. Three things it refuses to blur:

1. **Stub runs say so.** If a stub tier or stub Jev produced any part of the run,
   the summary says so in the first line and marks the run as stubbed. Nothing
   can be mistaken for real model output.
2. **Re-pricing is labelled as re-pricing.** The `repriced_usd` line takes the
   tokens this run actually used and multiplies them by the list price of each tier
   that can run a step (a `judge_only` tier is left out, because no configuration
   could have had it write these tokens). It is a counterfactual about price, not a
   prediction of what another model would have written, and it is printed as such.
3. **Jev's own spend is separated** from model spend, including the System One
   tokens that decision steps consume, and the currency is whatever your price table
   is denominated in, printed on every line rather than assumed to be dollars.
4. **Retries are visible.** Same-tier retries, error retries and escalations are all
   counted per step, because "it worked" and "it worked after three tries" are
   different facts about a tier.

A real run from this repo, with live System One and stub tiers (so no second key was
needed), took 3.3 seconds and cost **$0.0001** in 7 System One calls, and the gate scored
the placeholder deliverables at p=0.02, correctly escalating both. A later run with real
tiers as well cost **$0.0002** in 5 seconds and produced working code.

## CLI

```
run <task>      plan and execute; --dry-run stubs everything, --json emits the whole run
eval            pair cheap against strong across a task set and sweep the gate threshold
browse <goal>   drive a browser toward a goal, one typed decision per step
computer <goal> drive the macOS desktop toward a goal, one typed decision per step
plan <task>     show the decomposition without executing it
check           validate the config and report which keys are present
demo            a bundled dry run with no keys and no network
selftest        run the offline test suite
```

`browse` flags: `--url URL`, `--text STRING` (the string to type when an action needs one),
`--writer TIER` (compose text per field, gated by Jev), `--act` (without it, nothing is
touched), `--steps N`, `--min-confidence F`, `--headed`, `--json`, `--dry-run`,
`--ledger PATH`, `--quiet`.

`computer` flags: `--app NAME` (bring it to the front first), `--text STRING`, `--writer
TIER`, `--act`, `--steps N`, `--min-confidence F`, `--check-permissions`, `--json`,
`--dry-run`, `--ledger PATH`, `--quiet`.

`eval` flags: `--tasks PATH`, `--pairs N`, `--judge TIER` (who grades), `--strong TIER`
(which tier is the strong side), `--json`, `--out PATH`, `--dry-run`, `--quiet`.

Useful flags: `--planner llm|heuristic`, `--tier NAME` (pin one tier, skipping routing;
a `judge_only` tier is refused here), `--candidates N`, `--no-verify`, `--max-cost USD`,
`--ledger PATH`, `--quiet`. `task` may be `-` to read from stdin.

## Tests

`make test` runs 279 offline checks and needs no keys and no network:

- config validation, `${VAR:-default}` interpolation, tier ordering and escalation order,
  and the `judge_only` invariants: out of the ladder, out of the router's options, skipped
  when declared in the middle, refused as the first tier or the planner tier or `--tier`
- the eval's configuration: an explicit `strong_tier`, a `judge_only` tier never becoming
  one side of a pair, and the report naming both sides and saying whether it self-graded
- the Jev client against a **local HTTP server**: exact request payload, ordered
  score criteria, bearer header, 401 not retried, 529 retried then raised,
  non-object bodies rejected, long state clipped and marked
- planner JSON robustness: fenced blocks, prose around the JSON, bad kinds,
  duplicate ids, a Choice with one option degrading to Noul, step hoisting
- the loop: Jev-executed decision steps, the verification gate, escalation once,
  keeping the cheaper output when the strong one is worse, the full policy ladder
  (same-tier retry, error retry, retry caps, budget guard blocking both), candidate
  drafting and selection, routing fallback when Jev fails or returns nonsense, tier
  failure escalating instead of dying, budget stop marking the rest skipped, context
  propagation between steps, per-tier cost attribution and re-pricing
- provider clients against a **local HTTP server** for both wire protocols: request
  shape, bearer versus `x-api-key` headers, JSON mode, per-call token overrides,
  truncation detection, and an empty answer being reported as an error rather than
  passed off as a deliverable
- the eval itself: corrupt deterministically, blind un-mapping of a judge's verdict
  across randomized slot order, position-bias measurement, threshold sweep arithmetic,
  the refusals above, and a failed judge or generation being counted instead of fatal
- the local engine: question translation both ways, instruction synthesis when a caller
  omits one, answer normalisation, zero cost, visible state clipping, a refused question
  surfacing as a typed error, the missing-package message, and the checkpoint-resolution
  bug where a directly built config would send a hosted model name to Hugging Face
- the browser tool: options that name their target and stay mutually exclusive, action
  parsing, the dry run touching nothing, the two-no-op guard, the confidence floor, an
  off-menu choice refused rather than acted on, a failing engine stopping with its reason,
  the step ceiling, config-derived bridge settings reaching the session, and one real run
  against a local page fixture that clicks the right control and reveals the text it was
  after
- the writer: draft cleaning (one line, no wrapping quotes), a passing gate returning the
  draft with its verdict, one retry then an honest refusal, a provider error reported
  rather than raised, a failed gate stopping before a second draft is bought, and the
  spend booked under the writer's own tier in the ledger
- the desktop tool: field and press options that stay mutually exclusive, app switches
  that exclude the frontmost app, action parsing, every loop guard (two no-ops,
  confidence, ceiling, off-menu, engine failure), the `allowed_apps` guard by name and by
  bundle id, dry run touching nothing, ledger events, the writer composing a `set_value`
  and a failed gate stopping the step, a real-desktop test that is read-only and skips
  itself without the Swift toolchain or the Accessibility grant, and the config section's
  validation (bad actor, unknown or judge-only writer tier)

## Known limits

- **Jev cannot generate.** If a step needs prose, code, or a diff, a model does it.
  System One's value here is routing, judging, and answering closed questions.
- **The heuristic planner is deliberately dumb.** It exists so `--dry-run` and the
  tests are deterministic, and it splits on `" and then "`, `"; "` and newlines.
  The real path is `--planner llm`. The bundled demo shows the shape of a run, not
  a good decomposition.
- **The gate is only as good as `criteria`.** A vague criterion produces a vague
  probability. This is why the planner is instructed never to leave `criteria`
  empty, and why the fallback plan writes one for you.
- **Reasoning models make unreliable cheap tiers.** Measured on this machine: the same
  prompt produced 121, then 2994, then 1126 reasoning tokens across three identical
  calls to `deepseek-flash`, because thinking is charged to the same output cap as the
  answer. A tight cap therefore fails a small fraction of calls with an empty reply.
  The provider reports that as an error, `error_retries` retries the same tier, and the
  eval counts it. The fix is a generous cap plus the retry, or a non-reasoning model as
  the cheap tier. It is not only the cheap tier: at `planner_max_tokens = 2048` the
  planner fell back on 5 of 16 tasks, and at 6144 and then 12288 the *strong* tier still
  returned empty answers on the hardest steps. Any reasoning model in this loop needs a cap
  sized for its thinking, and the caps only bound the answer, so an unused cap is free.
- **The gate barely discriminates, which caps what any threshold can buy.** Measured on
  28 and 26 judged pairs: the gate's mean probability on cheap outputs the judge called
  sufficient was 0.88 (self-judging) and 0.86 (outside), and on the ones it called
  insufficient 0.86 and 0.89. One gap positive, one negative. A threshold on a signal like
  that is a cost dial, not a detector: raising it catches more shortfalls mainly by
  covering more steps. The reliable signal is corruption, where real outputs score 0.79 to
  0.84 against 0.34 to 0.36 for truncated ones.
- **A high `escalate_below` quietly turns the cascade off.** At 0.90 the loop escalates
  57 to 62% of steps and half of those escalations are wasted, so a step costs $0.0077 to
  $0.0095 against $0.0107 to $0.0141 for sending everything to the strong tier: a 16 to 20%
  saving, in exchange for the complexity. The saving is real but it is not the 3 to 5x the
  ladder suggests at lower thresholds.
- **Two judges, two answers, about a quarter of the time.** The self-judging and outside
  runs agreed on 13 of the 17 pairs both ruled on (76%). A recommendation that holds across
  both judges is worth more than one from either alone, which is why both were run.
- **A small eval is a small eval.** The sweep is only as trustworthy as the number of
  judged pairs behind it, which is why the report states that number and refuses to
  recommend below `min_judged_pairs`.
- **One escalation per step.** No retry loops, no cost-aware search over tiers.
- **The desktop tool drives the frontmost macOS app through its Accessibility tree.** That
  is a browser's gift extended to desktop apps, and it has the desktop's own absences: an
  app that draws its interface in a canvas (games, some Electron apps) exposes little or
  nothing to the tree; only the frontmost app is read, one app at a time; and the
  Accessibility grant is attached by macOS to whatever process runs the bridge (usually
  your terminal), so the grant follows the terminal, not this repo. No Screen Recording is
  requested and no pixels are read, by design.
- **The browser tool cannot write by itself.** Composing a search query or a message needs
  a text model, and neither decision engine generates text. Pass `--text` with the string
  to type, or set a `writer_tier` so the cheap tier composes it behind a Jev gate — the
  loop itself never invents words, and the ledger says which of the two happened.
- **The writer's gate is unmeasured at scale.** A draft is one line by construction
  (fields, not essays), and its gate fires at 0.6, deliberately below the agent's measured
  0.90, because "would typing this serve the goal" is a much easier question than "is this
  step's output done". Neither number has an eval behind it yet; the `write` events in the
  ledger are what a measurement would read.
- **An option list is a finite thing.** Controls are offered up to `[browser] max_controls`
  and the choice is a closed question, so a page whose target is the 40th link on a long
  list will be missed unless that link is near the top of the list. Off-screen controls are
  labelled rather than dropped, which helps, but it is not a search.
- **No tools yet.** Steps produce text; nothing edits files or runs commands. That
  is the natural next layer, and the ledger is already shaped for it.
- **The live path has been run, but not widely.** A real run on this machine cost $0.0002
  in 5 seconds for two steps: real planning, real routing, a System One answered decision
  step, real `deepseek-flash` output, real gate verdict. It has not been exercised on a long
  multi-step task, on a real repository, or with a tier failing mid-run outside the
  scripted tests.

## Next steps, in the order that makes sense

1. **Sharpen the step criteria until the gate discriminates.** This is now the measured
   bottleneck: with a discrimination gap of -0.03 to +0.11 the threshold cannot aim at hard
   steps, only cover more of them. The eval already prints the gap and warns when it is too
   small to be a signal, so the next measurement is criteria-first: rewrite the prompt the
   planner uses for `criteria`, re-run, and watch that one number rather than the
   recommendation.
2. **Choose the cheap tier by measuring it, not by its price.** Done once (runs G and H) and
   the answer was uncomfortable: `step-3.7-flash` is 4x cheaper per token than the strong tier
   and insufficient 25% of the time against `deepseek-flash`'s 19%, which at a 0.90 threshold
   cancels the saving entirely. The next measurement worth running is several cheap candidates
   side by side on the same pairs, scoring insufficiency rate against price per step.
3. **Re-run the eval on a bigger, harder task set.** The judged-pair count is past the floor,
   but the insufficient side is 2 to 5 samples depending on the run, and a 16-pair budget on
   these models only reaches 6 to 8 of the 16 tasks because the planner decomposes so finely.
   More pairs, not more tasks, is what would firm up the recommendation.
4. **Turn on drafting** (`candidates = 3`) for cheap tiers. Three cheap drafts plus
   a Jev pick is often cheaper than one strong draft, and the pick is logged. It is also a
   way to attack the real problem: with 81 to 89% of steps needing no strong model, the
   money is in cheap-tier reliability, not in escalation.
5. **Wire it into Reasonix** as an MCP server plus a skill, so an agent session can
   hand a decomposed task to this loop instead of doing every step on one model.
   Reasonix's own `triage_model` slot is the same idea for its internal
   classifications.
6. **Give the text loop the tools it is still missing** (read a file, apply a patch, run a
   test) and let the gate verify the result instead of the text. With tools, "insufficient"
   stops being a judgement call and becomes a failing test, which is the only way the gate
   gets a signal it cannot argue with. `browse` is the first of these: the pattern to copy is
   perceive deterministically, decide with one typed question, execute, record.
7. **Let a `run` plan include a browser step**, so a task can say "look this up and write the
   summary" and get both halves from one loop and one ledger, instead of running `browse`
   and `run` separately.
8. **Give the desktop tool its first measured run.** The browser's numbers above are
   measured; the desktop tool is built and tested but has no recorded run yet. The first
   measurement worth keeping: a small set of real goals on real apps, steps to done,
   confidence trace, engine cost per goal, and the failure modes the AX tree could not
   see. The same measurement repeated with `[computer] actor = "laya"` would say whether
   the hosted engine's action-selection edge holds off a browser.

## Layout

```
jev_cascade/
  config.py     TOML loading, validation, tiers, prices, env interpolation
  jev.py        System One client (choice / noul / score) plus an offline stub
  providers.py  OpenAI-compatible tier client, mock and scripted providers
  planner.py    decomposition, step typing, JSON robustness
  agent.py      the loop: route, execute, verify, retry, escalate, record
  laya.py       the local decision engine, behind the same protocol as the hosted one
  bridge.py     the JSON-line bridge client both tools share
  browser.py    browser use: page state to typed options, one action per step
  computer.py   desktop use: the Accessibility tree through a Swift bridge, same loop
  writer.py     a cheap tier composes typed text, gated by Jev, booked in the ledger
  eval.py       pair building, blind judging, threshold sweep, recommendation
  ledger.py     per-step records, JSONL events, honest summary
  cli.py        run / eval / browse / computer / plan / check / demo / selftest
tools/browser_bridge.mjs   the Playwright session the browser tool drives
tools/macos_bridge.swift   the Accessibility-tree reader the desktop tool drives
  testing.py    test doubles (ScriptedJev, StaticPlanner, FailingProvider)
tests/          279 offline checks, no keys and no network
evals/tasks.toml   the eval's 16-task set, difficulty interleaved on purpose
evals/results/     the measured eval reports quoted above, every run kept
config.example.toml
config.toml      your live config (gitignored)
```
