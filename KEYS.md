# codesearch API keys — status, quirks, unfixable nuances

Last verified: 2026-10-15. Live values live in `run` (gitignored). This file
documents BEHAVIOR only, never values. Placeholders in `run.example`.

## Key table

| Key | Required? | Live? | What breaks without it |
|---|---|---|---|
| `GITHUB_TOKEN` | Recommended | Yes (2 keys) | Anonymous 60/hr vs 5000/hr core. GHSA enrichment + code-search-bucket ops need it. Never route code search through GH (10/min bucket, auth-required). |
| `CONTEXT7_API_KEY` | No | Yes (2 keys) | Context7 lane 401s/degrades. Free tier cut to 1000 calls/mo Jan 2026 + 60/hr throttle — docs fallback now tries RTD/DevDocs/llms.txt FIRST to stretch quota. |
| `NV_KEY` | No | Yes | No embeddings → BM25-only hybrid rank. Code model `nv-embedcode-7b-v1` 4096d (deliberately different from websearch's general 1b model — documented, do not "fix"). |
| `SOFA_KEY` | No | Yes | `so_search action=sofa` fails; `stackexchange` path unaffected. |
| `LI_KEY` | No | Yes | Libraries.io 429s faster without. 60 req/min hard limit either way. Data is uncurated scrapes — prefer deps.dev + registries for facts. |
| `OSS_TOKEN` | No | Yes | Sonatype Guide scan auth. **Most fragile quota: free 500 credits/mo; legacy tokens die 2026-12-31.** OSV.dev fallback (keyless, no limits) fires automatically on 401/403/429/quota text — check `fallback: osv.dev` + `sonatype_error` in output. |
| `TAVILY_KEYS` | No | Yes | Single github-scoped lane in `search_all` skipped without. Free 1k credits/mo. |
| `CORE_API_KEY` | No | Yes | CORE lane weaker without, but CORE is demoted LAST in papers fallback anyway (no citation graph). Weakest lane by design. |
| `SE_API_KEY` | No | Yes (wired + sent) | Keyless 300/day/IP → keyed 10k/day/IP. **Fixed 2026-10-11: `backoff` field now honored** (`_se_get` sleeps per-method deadlines). Schema text corrected to "300 req/day" (was "300 req/min"). |
| `GITHITS_API_TOKEN` | Effectively yes | Yes | **PkgSeer dead without it** — `code_search`/`code_files`/`code_read` fail. Mitigation is jsDelivr-first (npm) + Sourcegraph lane, not more PkgSeer reliance. |
| `S2_API_KEY` | No | NOT SET (passthrough only) | Keyless S2 429'd live on 2026-10-11 AND 2026-10-15 (shared-pool contention). Free key = dedicated 1 RPS + batch-500 counts as 1 req. **Register at semanticscholar.org/product/api and export it — highest papers ROI.** Papers fallback (arXiv→OpenAlex→CORE) covers outages, verified live under 429. |
| `OPENALEX_API_KEY` | No | NOT SET (passthrough only) | Keyless still works but polite-pool retired Feb 2026 (legacy bucket). Free key = 10x budget, sent as `?api_key=`. Register at openalex.org. |
| `GROQ_API_KEYS` | No | NOT SET | `search_all synthesize` + `code_expand` are **silent no-ops** in prod (`has_keys` guard). Code now defaults `synthesize=false` without keys + returns a clear notice when explicitly requested. Add keys or accept raw results. |
| `FIRECRAWL_KEYS` | Dead | REMOVED from run.example | Zero `.py` references (verified). Do not re-add. |

## Unfixable / accept-as-is

1. **Sonatype quota fragility.** 500 free credits/mo is a hard vendor ceiling; OSV.dev fallback is the mitigation, not more Sonatype reliance. Legacy tokens die 2026-12-31 — expect one forced rotation.
2. **S2 keyless contention.** Shared anonymous pool 429s are routine in 2026, not an outage. Only a free key fixes it (register, don't retry harder).
3. **PkgSeer single-vendor gate.** No public status page; treat as unmonitored. jsDelivr-first + Sourcegraph lanes are the hedge.
4. **DeepWiki/CodeWiki fragility.** Both unofficial scrapes; CodeWiki hits an internal Google batchexecute RPC that can vanish unannounced. Kept behind timeouts; grep.app + Sourcegraph reduce analyze-time dependence.
5. **Libraries.io staleness.** Alive under Sonar but data is uncurated scrapes. Keep for source-rank/dependents; never for version facts.
6. **SearchCode 429 (E5 fixed).** Now honors Retry-After with exp-fallback, cap 30s, 3 retries (mirrors `_s2_backoff`). Generous free POST API otherwise.
7. **`run` holds live keys in plaintext** (gitignored + pre-commit tripwire). Never `git add -f run`. Tripwire false-positives on `YOUR_KEY`-style placeholders — use `--no-verify` for placeholder-only diffs.
8. **Deferred-import discipline.** Two live `UnboundLocalError` incidents (vulns-scan 2026-10-11, ping-shadow 2026-10-15) from `from X import Y` inside the dispatch function shadowing module-level names. Rule: ALL imports at module level in `server.py`. The `except Exception: pass` guards on the four optional-enrichment blocks (libraries_io, vulns, ghsa, scorecard) are INTENTIONAL — fail = field absent, base result intact.
