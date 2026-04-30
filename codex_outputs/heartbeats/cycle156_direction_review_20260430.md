Trajectory: **still on-axis, but low-ceiling.** g188 killed static OT bridging but exposed a real interface prior: exact string-matched trained rows carry almost all of g181b. That is worth decomposing, so **let g191 finish**. But if the queue simply goes g191 → g192, it risks becoming “shared-vocab row reuse is real,” not a breakthrough.

**Active queue §0.1 expected uplift: 5/10.** g191 is necessary; g192 is useful validation but mostly ceiling-limited. Even PASS_CONTENT + 28-layer PASS only gets you around **5.5-6.0**, and the 84% overlap critique remains.

**Reorder next queued experiment:** yes. After g191, do **not** automatically fire g192 as the next main move. Insert a **token-row compiler** first.

Concrete change: train a small row generator using exact matched GPT-2/Qwen token strings as supervision: token bytes + frequency/context stats → Qwen-format embedding row. Hold out matched rows, generate rows for held-out/unmatched tokens, then run GPT-2-tokenizer Qwen shell with **no copied target rows**. PASS bar: generated-row arm beats scratch by **≥ +0.30 nats**, positive 3/3 seeds, and shuffled/frequency controls ≤ +0.10.

Why: this turns the 84% shared-vocabulary weakness into a construction law. g192 asks “does copying survive depth?” The compiler asks “can we synthesize the interface prior?” That is much closer to §0.1: non-obvious, useful, and harder to dismiss as lexical overlap trivia.

