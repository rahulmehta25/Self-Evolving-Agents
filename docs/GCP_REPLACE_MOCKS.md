# Replacing Mocked Components with GCP (Using Your Credits)

The production path supports **Judge (Vertex Gemini)**, **Vertex AI Search (retrieval)**, and **real tool APIs** via configurable URLs.

---

## 1. Judge (LLM-as-Judge) — ✅ Vertex (default when available)

**Enable:** Judge uses **Vertex AI Gemini** when the Vertex SDK is available (same as agent). Set `EVOAGENTBENCH_JUDGE_LLM=vertex` to force, or `EVOAGENTBENCH_JUDGE_LLM=mock` to force mock.

- **CLI:** `--judge vertex` or `--judge mock`
- **Auth:** Same ADC/project as the agent.
- **Cost:** One extra Gemini call per run when judge is vertex.

---

## 2. Tools (get_weather, retrieval, etc.) — Options

**Before:** `_mock_tool_execute` returns fixed results (e.g. get_weather → sunny).

**Options to replace with GCP/credits:**

| Tool type    | GCP option                         | What you need to do |
|-------------|-------------------------------------|----------------------|
| **Retrieval / RAG** | **Vertex AI Search** (Discovery Engine) | Enable Vertex AI Search API, create an app and data store, index your docs. Agent runner would call the Search API and log snippets/source/span for the citation checker. |
| **Retrieval (simple)** | **Gemini + context** | No new API: pass task context (or a doc) into a single Gemini call that “answers from document.” Less scalable than Search; good for one-doc benchmarks. |
| **get_weather / external APIs** | **Cloud Functions** or **Cloud Run** | Deploy a small HTTP endpoint that calls a weather API (or other API); agent runner calls your endpoint. You pay for invocations + any external API. |
| **Custom logic** | **Cloud Functions** | Implement tools as functions; runner calls them via HTTP. |

### What to choose

- **Only Judge real:** Set `EVOAGENTBENCH_JUDGE_LLM=vertex`. No extra setup; uses existing Vertex credits.
- **Judge + real retrieval:**  
  - **Quick path:** Add a “retrieval” tool that calls Gemini with the task’s `context` (or a single document) and returns an answer + snippet. No Vertex Search setup.  
  - **Full path:** Use Vertex AI Search; you enable the API, create a data store, and we wire the runner to call it and log source/snippet/span for citations.
- **Weather / other APIs:** You provide or deploy a small backend (e.g. Cloud Function) that the runner can call; we’d add a small “tool dispatcher” that routes tool names to your URLs.

### What I need from you for tools

1. **Retrieval only (simple):** Nothing — we can add a “Gemini-in-context” retrieval tool that uses the task’s `context` and returns a structured snippet (no new GCP products).
2. **Retrieval with Vertex AI Search:** You (or someone with project access) enable **Vertex AI Search API**, create a **data store** and index your benchmark docs; then share the **data store ID** (and region if not default). I can then wire the runner to call Search and log snippets/source/span.
3. **Weather or other APIs:** Either a public/sandbox API key (or URL) for the service, or a Cloud Function/Cloud Run URL that the runner can call. I’ll wire the runner to that URL for the corresponding tool name.

---

## Summary

| Component | Status | Your action |
|-----------|--------|-------------|
| **Judge** | Real Vertex available | Set `EVOAGENTBENCH_JUDGE_LLM=vertex` or `--judge vertex`. |
| **Tools** | Still mock | Choose: (A) simple retrieval via Gemini+context, (B) Vertex AI Search (you enable API + data store), or (C) Cloud Function/URL for weather or other APIs. |

All of these use your existing GCP project and credits; Judge uses Vertex Gemini only; tools use Vertex Search and/or Cloud Functions/Cloud Run as you choose.
