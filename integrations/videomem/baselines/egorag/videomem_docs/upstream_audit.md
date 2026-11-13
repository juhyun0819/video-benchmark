# EgoRAG upstream audit

Audited checkout: `EvolvingLMMs-Lab/EgoLife`, upstream commit `7a97157908757cc898c26835b718653055ecc5f5` (2025-03-19). Paths and line numbers below refer to that upstream revision, relative to `EgoRAG/`. No applicable `AGENTS.md` was found in the repository ancestry or the EgoRAG subtree. This audit changes only this document.

The implementation supplies reusable caption indexing, hierarchical time selection, text retrieval, evidence extraction, and answer prompts. Its shipped CLI is **caption-only at answer time**, its ordinary query path consumes annotation-provided `keywords`, and its time boundaries are insufficient for a strict historical-memory evaluation. It should be described as the upstream implementation with documented adaptations, rather than an exact reproduction of the paper's experiment.

Validation comprised source inspection and isolated execution of selected upstream functions with in-memory database/LLM stubs. No packages were installed, model weights downloaded, credentials inspected, or paid inference APIs called. This is not an end-to-end model-quality or benchmark-accuracy result.

## Create: actual call trace

Two entry points converge on the same text collection:

```text
create_database.py:main
  Chroma(name=db_name)
  RagAgent(database_t=..., name="A1_JAKE", video_base_dir="Egolife/train")
  RagAgent.create_database_from_json(json_path)
    JSON caption -> sentence strings -> OpenCLIP text embeddings -> collection.add
  gen_event(db_name, diary_dir)

main.py:main --stage create
  enumerate participant/day files; load YAML; import and instantiate caption model
  RagAgent.create_database_from_video(...)
    parse DAY and filename start times
    get_video_mapping -> parse_video_map
    add_to_db_t -> model.inference_video(...) -> caption
    caption -> sentence strings -> OpenCLIP text embeddings -> collection.add
  gen_event(db_name, diary_dir)
```

`RagAgent.py:196` expects a JSON **array** of caption entries, although the README presents a single entry example. Required entry fields are `date`, `start_time`, `end_time`, and `text`; `video_path` is optional. Sentence IDs are `DAYn-start-end_sentenceIndex`; metadata contains numeric day, integer HHMMSSFF start/end, and video path. JSON creation splits with `re.split(r"(?<=[.!?])\s*", text)`; video creation splits on periods and appends periods (`RagAgent.py:117`). All sentences from a clip share its complete interval. JSON ingestion reads existing IDs and skips the whole entry when any matching document has content; this is not a reliable partial-ingestion repair operation.

The video entry point uses the first-person caption prompt in `main.py:70` and its egocentric system prompt at line 71. It really invokes `model.inference_video` in `RagAgent.py:105`; it does not create captions from filenames. However, `get_video_mapping` uses consecutive filename starts as clip endpoints and iterates only through `len(video_time_list)-1` (`RagAgent.py:75`). Consequently, the last file each day is omitted, a day with one file produces zero samples, gaps are treated as interval duration, and a fixed 30-second segmentation is not enforced. Use an explicit segment manifest with actual durations for adapted data.

`caption_args` is accepted and passed down but never consumed by `add_to_db_t`. A YAML `caption.fps` value therefore does not configure frame sampling in this path. `main.py` enumerates videos and initializes the caption model before checking stages, even for text-only query/answer runs (`main.py:40`, `55`).

## Hierarchical summaries: actual structure and prompts

`gen_event` is a three-level pipeline (`egorag/utils/gen_event.py:480`):

| Level | Construction | LLM helper / prompt | Storage |
| --- | --- | --- | --- |
| L1 | Group sentence documents into nominal ten-minute windows, selected by document start | `call_gpt4`; `system_message_min` | `<db>_l1/l1_dayN.json` |
| L2 | Group L1 items into roughly one-hour batches measured from the first item's start | `call_deepseek`; `system_message_hour`, under 500 words | `<db>_l2/l2_dayN.json` |
| L3 | Summarize all L2 items for a day | `call_gpt4`; `system_message_day`, around 1,500 words requested, 2,048 token cap in Azure path | `<db>_l3/l3_dayN.json` |

These prompts request chronological, first-person, factual event summaries that retain decisions, actions, participants, purpose and method while removing redundant conversation. L1 has a DeepSeek fallback for some raised exceptions, but an API helper's literal `"error"` return is truthy and can be stored as a summary. The last L1 batch does not have the same exception fallback.

All levels store `generated_text`, `date`, `start_time`, `end_time`. Each level also writes `merged_day_1to7.json`; generation hardcodes days 1–7. Query consumes the merged **L2 and L3** files (`main.py:99`), not L1 directly. The summaries are JSON text passed to LLM selection; they are not embedded or indexed in Chroma. The hierarchy is therefore a language-model time selector followed by sentence-vector retrieval.

Critical implementation details:

- `get_doc` returns collection order; summary generation never explicitly sorts by time (`Chroma.py:133`, `gen_event.py:217`). The chronological assumption is unverified.
- L1 uses `start_time < current_window_end` to include the **entire** caption, without checking its end. Its recorded boundary may precede source content. `minutes_to_time` first truncates minutes to an integer, then computes a fractional component that is always zero (`gen_event.py:104`), losing seconds as well as frame precision.
- L2/L3 inherit these boundaries and record first/last item bounds rather than the minimum start and maximum source end. No source-ID provenance is retained to validate summary visibility later.
- L1/L2 merged-file discovery includes every `.json`, including an existing merged file on rerun (`gen_event.py:250`, `388`). This can duplicate cached content. L3 skips valid per-day files but does not append their contents to its `results`; a fully cached rerun overwrites the merged L3 output with `[]` (`gen_event.py:415`, `420`, `471`).
- Temporary paths use `output_file.split(".")[0]`. With the generated `./...` path this evaluates to an empty string, so the actual temporary file is `_tmp.json` in the working directory, rather than a sibling of the output. The sequential path may still run, but concurrent executions collide.

## Query: two distinct paths

The common route is `main.py -> query_all -> single_query -> query -> parse_query` (`RagAgent.py:400`, `496`, `502`, `525`). `parse_query` reads question, choices A–D, `query_time`, and **the annotation's `keywords`**. Missing `query_time` raises because the default is `None` and `.get` is then called; missing keywords becomes the literal `UNKNOWN_QUERY`.

### Ordinary questions

1. `filter_event_data` includes prior-day L3 summaries and current-day L2 summaries whose recorded end is at or before query time. It generates a current partial summary from captions with `start >= partial_start` and `end <= query_time` (`RagAgent.py:260`, `284`). This is a useful intended boundary but depends on correct metadata. It trusts ordering, assumes `hour_docs[0]` exists, and extracts a day with `int(date[-1])`.
2. `get_range` calls `call_gpt4` with `get_day_prompt` and the visible summary list. It expects one `[day]`; failure to parse defaults to the query day. If a different day is chosen, it loads **all** hourly summaries for that model-selected day, without checking that the day is historical (`RagAgent.py:362`).
3. Another `call_gpt4` with `get_hour_prompt` selects one start/end range. The parser expects two newline-separated bracketed eight-digit times. A missing line raises. The returned range is not validated for day membership, ordering, supported evidence, or query cutoff.
4. `Chroma.custom_query(query_texts=[annotation_keywords], n_results=3, filter_first=True)` filters documents to that day and full containment in the chosen interval, computes text-vector cosine similarities, and returns three **sentence rows** (`RagAgent.py:472`, `Chroma.py:154`). Multiple hits may belong to one clip; there is no unique-clip aggregation or diversity pass.

Only one day and one range are selected, so the general path cannot directly collect evidence distributed across several selected days. There is no ordinary-query keyword generation step; options influence the final answer but are not included in the day/hour selection prompts.

### Questions containing `last time` or `the last`

This branch is selected by English substrings in the question (`RagAgent.py:402`), not the annotation's `last_time` flag. Other forms such as “last used” do not necessarily enter it.

1. `query_prompt` asks `call_gpt4` for short action/object keywords and an upper search time. It uses the question, choices, and query timestamp; it asks for a time before the question and, for non-time options, roughly five minutes earlier (`prompts.py:64`). This is the only implemented keyword rewrite.
2. A Chroma-native query retrieves up to 50 sentence rows across prior days plus the chosen day's rows ending before the **model-generated** bound (`filter_first=False`, `RagAgent.py:435`). There is no independent intersection with the original cutoff.
3. `process_query_results` sorts hits by day/end and expands each by one sentence index on either side. `get_ids` asks an LLM to choose the most recent relevant result (`agent_fun.py:4`). It then loads that selected ID with one neighboring sentence on each side.

The ID parser accepts only one digit of sentence index (`agent_fun.py:62`), so `_12` can be parsed as `_1`; it also does not validate that the chosen ID was actually offered. If no time range was parsed, `search_time` remains unset and is later read in the result dictionary (`RagAgent.py:464`).

The top-50 search followed by an LLM ID choice is **reranking**, not an additional retrieval pass. No iterative query rewrite, evidence-conditioned follow-up search, search-budget loop, or retrieval retry after empty evidence exists in this revision. Saving query-result JSON is supported, but a dedicated GPT-4o-mini rewrite cache/workflow is not present.

## Embeddings and similarity

`Chroma.__init__` creates a persistent client at `./chroma_db/<name>` and configures `OpenCLIPEmbeddingFunction()` without explicit encoder/checkpoint arguments (`Chroma.py:36`). Both ingestion paths pass sentence **strings** to it; query passes text. `ImageLoader` is attached to the collection but no image embeddings, frame URIs, or image ingestion calls are used by these paths. `RagAgent.database_i` is assigned but never used. There is no implemented visual/text weighted fusion.

For ordinary queries, `custom_query` first fetches all rows matching metadata and computes explicit cosine similarity in NumPy. It handles only the first query vector for ranking and returns flat arrays. The last-time branch calls the collection's native query, returns nested arrays, and lets the collection choose its configured distance; this file does not set an explicit metric. Do not claim these two scoring paths are identical. `process_query_results` expects the nested shape and `extract_evidence` expects the flattened/adapted shape.

`get_caption` generates IDs that vary only the final sentence index (`Chroma.py:69`, `92`). Its “neighbors” are sentences in the **same interval prefix**, not adjacent video clips. It returns all neighbor documents but only the requested ID and the first returned metadata row. It does not preserve a full row-wise provenance mapping or explicitly sort by sentence index. Same-clip expansion is useful, but adapted evidence should track actual source rows and deduplicate clip reads.

## Answer: prompts and actual video use

```text
main.py --stage answer
  load saved query result or use query-stage result
  extract_evidence(question, result, modality="text")
    for each retrieved sentence ID:
      get_caption(id, n_result=10)  # within same clip prefix
      construct timestamped caption
      call_gpt4(text_evidence_prompt)
      parse "I can provide evidence. Evidence: ..."
  get_answer(question, query_time, choices, evidence_cards, caption_results)
    nonempty cards -> with_evidence prompt -> call_gpt4
    empty cards -> without_evidence caption prompt -> call_gpt4
    parse first standalone A/B/C/D
  calculate_accuracy using metadata.answer
```

The decisive CLI call is `main.py:151`: `modality="text"`. Thus selecting EgoGPT or a video model in YAML does **not** make the shipped answer stage revisit video. The captions are expanded and sent to `call_gpt4`, then the final answer is also generated through `call_gpt4`. If all evidence cards are rejected, the original retrieved captions are still supplied by the fallback answer prompt; no second search occurs.

`extract_evidence` itself defaults to `modality="video"` (`RagAgent.py:555`). When invoked that way, it calls the configured model's `inference_video(video_path, video_start_time, start_time, end_time, video_evidence_prompt)` at line 603, which is a real video revisit. It still builds expanded caption context; if video evidence is rejected, `get_answer` can fall back to those captions. The main program exposes no CLI switch for evidence modality.

Model inspection confirms actual pixels in the optional path: `models/azure.py:38` opens the local video with Decord, extracts frames, resizes to 384×384, JPEG-encodes them, and sends `image_url` contents in an HTTP request. `models/qwen2_vl.py:63` opens the video and turns sampled frames into PIL images; its calculated FPS sample list is then replaced with 30 uniform samples in the inference path. Therefore sampling behavior depends on the adapter rather than just YAML settings.

The EgoGPT adapter also loads audio (`models/egogpt.py:119`), but its slice uses raw HHMMSSFF values multiplied by 16 kHz (`start_time * target_sr`) without conversion to seconds or subtraction of video start. Typical daytime timestamps yield out-of-range, empty audio slices. Actual audio-aware reproduction requires this correction and verification; the presence of `<speech>` tokens alone is not proof of meaningful audio input.

The evidence parser relies on exact English phrases and a non-DOTALL regex, making it brittle for multiline output. The final prompt requests a letter only. A literal API `"error"` produces a random choice (`RagAgent.py:658`), rather than an explicit failure. Query/answer loops catch errors and omit failed questions; `calculate_accuracy` divides only by retained answers. Reproduction metrics must preserve every selected question and report coverage and failures, not inherit that denominator silently.

## Paper versus this checkout

The comparison uses [arXiv:2503.03803v3, Sections 4.2 and 5](https://arxiv.org/html/2503.03803v3#S4.SS2), also checked against v1. The [version page](https://arxiv.org/abs/2503.03803) identifies v3 as the latest revision and describes its change as affiliation correction. The relevant technical description is unchanged in the passages inspected.

The paper describes caption memory with hourly/daily summaries and a general visual/text relevance formula. Its experimental account uses text retrieval, three 30-second clips, video-model evidence, GPT-4o final answers, and GPT-4o-mini re-querying with stored results. It also explicitly identifies single-pass retrieval without iterative refinement as a limitation. The statement that text-only retrieval sets lambda to zero is inconsistent with its displayed formula, where lambda weights textual descriptions. Treat the experimental prose as intent, not evidence of a missing implemented fusion model. [Paper, Sections 4.2–5](https://arxiv.org/html/2503.03803v3#S5)

Concrete implementation differences, established by the source above:

| Topic | Pinned implementation |
| --- | --- |
| Retrieval representation | OpenCLIP-encoded sentence text; dormant image database field; no fusion scorer |
| Retrieval unit | Top three sentence rows in the ordinary path, with possible duplicate clips; last-time path retrieves 50 and reranks to one |
| Segmentation | Consecutive filename timestamps, omitted final file; no enforced 30-second manifest |
| Summary hierarchy | Extra ten-minute L1; L2 uses DeepSeek; L1/L3 use the GPT helper |
| Video evidence | Available in an optional method branch, but CLI explicitly chooses text evidence |
| Final/rewrite model identity | Azure endpoint determines deployment externally; non-Azure helper literally requests `gpt-4` |
| Rewrite machinery | Special-case keyword extraction only; no named GPT-4o-mini model or general rewrite/cache loop in this code |

`call_gpt4` also invokes the legacy `openai.ChatCompletion.create` interface (`util.py:78`) while `pyproject.toml` pins OpenAI Python 1.55.3. The non-Azure path is incompatible with that installed SDK generation; it catches exceptions and returns `"error"`. Its `max_tokens` is hardcoded to 2200 even when the caller supplies another value. Azure requests have no timeout. The helper name is not reliable model provenance. The README's answer-only command additionally hits `query_json_name` before initialization (`main.py:127`); that variable is assigned only in the query stage.

## Ground-truth and temporal boundaries for adaptation

**Ground truth:** the ordinary upstream retrieval query reads annotation `keywords` directly. This is privileged retrieval supervision and must be removed from the public model input unless the run is explicitly labeled as a keyword-oracle experiment. Generate query text from the permitted question/options instead. `answer` is used for evaluation; `target_time`, `reason`, `need_audio`, `need_name`, `trigger`, and `last_time` are not read to build prompts in the inspected path. It would be inaccurate to claim that all those fields are fed to the LLM. However, `query_all` copies the entire raw QA object into output `metadata` (`RagAgent.py:513`), and the answer stage carries it forward. The adapted boundary should expose only ID, question, choices and query time to index/retrieval/model code; keep labels and annotation diagnostics in evaluator-only records.

**Temporal visibility:** filtering final hits is insufficient if a future-derived summary has already influenced day/hour selection. Every summary offered to a model must be derived solely from admissible source intervals. Do not relabel a whole summary's end to the cutoff: regenerate the partial summary from valid children, or omit it. Use source interval ends/provenance, not the nominal bucket label, to establish visibility.

The required controls are:

1. Convert DAY plus HHMMSSFF into a canonical comparable time with explicit frame-rate semantics. Define interval and equality behavior. For half-open source intervals `[start, end)`, `end <= cutoff` ensures no source frame at/after cutoff is admitted; a stricter benchmark policy should be applied consistently instead.
2. Filter valid source segments **before** summary construction or exposure. Record child IDs and the maximum true source end. Summaries spanning the cutoff are excluded or recomputed using only admissible children.
3. Restrict day choices to the offered historical days and intersect any model-selected interval with the original cutoff and data bounds. A model response must never enlarge access to future summaries, captions, or video. Preserve the same cutoff for every rewrite or additional search if an adaptation adds those features.
4. Enforce the boundary again before returning expanded captions and before reading video/audio. For a clip crossing cutoff, either exclude it or create a genuinely truncated input/caption; changing metadata alone does not remove future information.
5. Key caches by source content/provenance, model/prompt settings, participant and cutoff. Whole-day cached summaries cannot serve an earlier same-day query without valid source coverage.
6. Log actual source IDs, intervals, retrieval scores, selected ranges, summary provenance, model names and whether each evidence card came from text or decoded video. Keep annotation labels out of model trace payloads.

## Reuse decisions

| Component | Reuse recommendation |
| --- | --- |
| Caption/segment metadata schema and first-person caption task | Retain the concept; adapt timestamps and explicit segment manifests to the dataset |
| `get_day_prompt`, `get_hour_prompt`, evidence and answer templates | Reuse with attribution and a constrained parser/selector; enforce bounds outside the prompt |
| `gen_event` factual summarization instructions | Reuse the prompts; replace windowing, ordering, coverage metadata and cache merge logic |
| Chroma/OpenCLIP sentence retrieval | Reuse behind an injectable embedding/store interface if the encoder is available; record any replacement embedding as a methodological deviation |
| `get_caption` expansion idea | Retain same-segment context expansion with explicit provenance, stable ordering and clip deduplication |
| Video adapter interface | Reuse as an interface; use a validated reader and frame/audio interval conversion, and expose modality explicitly |
| Top-50 latest-event reranking | Optional upstream behavior; fix ID parsing, membership checks, cutoff enforcement and routing before use |
| API wrappers, stage CLI, error handling and metric denominator | Replace/fix; do not reuse unchanged for a reported experiment |

A local, no-API implementation may validate data plumbing, leakage controls and retrieval determinism with stubs or substitute components, but should report those substitutions. It cannot establish upstream LLM quality or paper-equivalent accuracy. If an adaptation supplies a general rewrite or additional-search loop, label it as added behavior; it is not present in this upstream revision.

## Executed isolated probes

The following observations were asserted successfully by extracting the relevant unmodified function ASTs and substituting in-memory LLM/database responses. Import-time model dependencies and network code were not executed.

| Probe | Observed result |
| --- | --- |
| `get_video_mapping` with one video start | Empty mapping |
| `get_video_mapping` with three video starts | Two mapped intervals; final video omitted |
| `parse_query` / ordinary `query` with `keywords="GT_ORACLE_KEYWORD"` | Exact sentinel passed as vector query text |
| DAY2 question; mock day selector returns DAY3 | DAY3 hourly text containing `FUTURE_SECRET` passed to the hour-selector prompt |
| Ordinary query using that DAY3 range | Database predicate requests DAY3 13:00–14:00 without intersection with DAY2 cutoff |
| `minutes_to_time(time_to_minutes(10103000))` | `10100000`, losing 30 seconds |
| L1 with source intervals 10:00:30–10:01:00 and 10:10:20–10:10:50 | Both source texts included, but summary end recorded as 10:10:00 |
| `get_ids` receives mock output ending in sentence suffix `_12` | Parsed ID incorrectly ends in `_1` |
| Temporary-path expression on `./events_diary/db_l1/l1_day1.json` | Prefix collapses to empty string, yielding working-directory `_tmp.json` |

These probes demonstrate boundary defects and control flow, not the behavior of a real language model. Source inspection additionally established the caption-only CLI, raw annotation metadata propagation, cached-summary merge issues, SDK mismatch, and video/audio adapter limitations described above.
