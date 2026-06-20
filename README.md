# ReCompress

Query-aware compression beyond deletion — built for The Token Company Compression Challenge (UC Berkeley AI Hackathon 2026).

## Overview

**bear-1.1** (The Token Company) compresses prompts by deleting low-value tokens character-for-character — fast and lossless-by-design, but blind to the query and unable to rewrite. **ReCompress** adds the two things deletion can't:

1. **Query-aware selection** — reads the question, drops irrelevant passages
2. **Dense rewrite** — densifies verbose-but-relevant prose

Then distills the query-aware compressor into a small model (Qwen2.5-3B) via recursive self-improvement (STaR-style).

## Structure

```
ReCompress/
├── token-company-prd.md       # Project PRD / plan
├── Token Company Prompt.pdf   # Challenge brief
├── archive/                   # Reference materials
└── src/                       # Implementation (coming)
```

## Status

Pre-build — scaffolding.
