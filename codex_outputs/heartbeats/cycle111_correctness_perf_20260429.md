Not fully clean.

1. **Stage1 cache path:** stage1 loads `results/cache/genome_182_features/teacher_texts.json` at `code/genome_182_triage_arena.py:2400`. The normal Qwen3 stage1 generator writes the same path at `2409-2410`. Good.

   But `generate_phase2_teacher_texts()` is **not** the stage1 path. Phase2/frozen eval writes `teacher_texts_{phase2_arch}.json` at `1679-1687`. If the running standalone process writes `teacher_texts.json`, stage1 will load it. If it writes `teacher_texts_qwen3.json` or `teacher_texts_falcon_h1.json`, stage1 will miss it and regenerate.

2. **Race condition:** if you launch stage1 only after PID 44380 exits cleanly, no race. If you launch while the writer is still dumping JSON, there is a real race because teacher cache writes are **non-atomic** direct writes, unlike `save_incremental()`. Stage1 can see an existing partial `teacher_texts.json` and crash on `json.load()`. Wait for process exit and stable file size.

3. **`--stage1-only` CLI:** yes, it skips teacher generation **if `teacher_texts.json` exists**. It does still load Qwen3 donor and build GPT2/Qwen vocab first, but teacher gen itself is skipped at `2401-2405`. Caveat: it does not validate cache length/provenance. Confirm `len == 8704` before launch.

4. **Windows risks:** main risks are partial JSON read and stale file handles. Encoding is okay: reads/writes use UTF-8 for stage1 cache. JSON dump default `ensure_ascii=True`, so Unicode text is escaped. File locking should be fine after process exit. Also no atomic cache write means interrupted teacher gen can leave a corrupt file that stage1 will trust and fail loading.

Bottom line: launch is safe **only if** final file is exactly:
`results/cache/genome_182_features/teacher_texts.json`, valid JSON, length `8704`, and PID 44380 has exited. 
