# Aegis — Defense-in-Depth Security for Tool-Using LLM Agents

> **The model can be fooled. It should not be able to act on that instruction.**

Aegis is a layered security system for tool-using Large Language Model (LLM) agents. It protects user inputs, retrieved content, proposed tool actions, and generated answers through input guards, data-protection checks, a quantum-kernel prompt-injection detector, tool-safety policies, grounded-output verification, and tamper-evident audit logging.

**Project scope:** Demo tools are simulated. Q-Gate runs on a quantum simulator. This project does not claim quantum advantage.

---

## Table of Contents

- [The Problem](#the-problem)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Request Lifecycle](#request-lifecycle)
- [Security Layers](#security-layers)
- [Grounding and Output Verification](#grounding-and-output-verification)
- [Q-Gate: Quantum-Kernel Injection Detection](#q-gate-quantum-kernel-injection-detection)
- [Simulated Tools](#simulated-tools)
- [Audit Logging](#audit-logging)
- [Evaluation and Testing](#evaluation-and-testing)
- [Technology Stack](#technology-stack)
- [Installation and Setup](#installation-and-setup)
- [Running Tests](#running-tests)
- [Example Attack Scenario](#example-attack-scenario)
- [Limitations](#limitations)

## The Problem

LLM agents can retrieve documents, process user instructions, and invoke tools such as email, file, and database functions. This creates several security risks:

- **Jailbreaks:** Users attempt to bypass the agent's safety rules.
- **Prompt injection:** Malicious instructions are embedded in retrieved documents or tool outputs.
- **Data leakage:** Sensitive information may be exposed through generated responses or outbound actions.
- **Unsafe tool use:** The agent may attempt unauthorized actions or use untrusted tools.
- **Hallucinations:** The model may generate claims that are unsupported by available evidence.
- **Insufficient auditability:** Security decisions may be difficult to investigate or verify.

A system prompt or a single classifier is not a complete security boundary. Aegis uses multiple layers so that detecting suspicious content and authorizing actions are separate responsibilities.

## Key Features

- Input normalization and risk classification.
- Multi-turn risk tracking and refusal handling.
- Retrieval access checks, sensitive-data scanning, canary detection, and taint tracking.
- Q-Gate prompt-injection detection using a quantum-kernel SVM.
- Classical RBF SVM comparison.
- Tool registration, argument validation, action limits, and confirmation policies.
- Single-use, HMAC-based action tokens.
- Grounded answer generation with passage citations.
- Citation validation and source-value checks.
- Optional natural-language inference (NLI) for claim verification.
- Output decisions to answer, prune unsupported content, or abstain.
- SHA-256 hash-chained audit logs.
- Evaluation tools for comparing configurations and measuring security outcomes.

## System Architecture

The system is organized around a protected agent workflow:

1. Input validation and risk assessment.
2. LLM reasoning and tool selection.
3. Tool authorization and simulated execution.
4. Retrieved-content scanning and data protection.
5. Prompt-injection detection.
6. Evidence-grounded answer generation.
7. Output verification and policy enforcement.
8. Audit recording and evaluation.

If available in the repository, the architecture diagram is located at:

`docs/aegis_architecture_flowchart.png`

## Request Lifecycle

1. **User input:** The chat interface forwards the user's message to the agent pipeline.
2. **Input guards:** The message is normalized and classified. Risk tracking and refusal rules may also apply.
3. **Agent processing:** The LLM generates an answer or proposes a tool call.
4. **Tool safety gate:** The proposed action is evaluated and assigned an `ALLOW`, `CONFIRM`, or `BLOCK` decision.
5. **Action authorization:** Allowed tool calls are checked against the relevant authorization and action-token requirements.
6. **Simulated execution:** The permitted tool executes in the controlled demonstration environment.
7. **Ingress and data protection:** Retrieved content is checked for access permissions, sensitive data, canaries, provenance, and taint.
8. **Q-Gate analysis:** Untrusted text is evaluated for possible prompt injection and may be passed, reviewed, or quarantined.
9. **Grounded generation:** The LLM uses vetted passages and their identifiers to answer the user's question.
10. **Output verification:** Citation validity, source support, claim entailment when available, and output-safety rules are checked.
11. **Final decision:** The system returns a supported answer, prunes unsupported claims, or abstains.
12. **Audit logging:** Relevant security decisions and events are recorded in the hash-chained audit log.

## Security Layers

| Layer | Responsibility |
|---|---|
| J1 | Normalize and classify incoming messages. |
| J2 | Harden prompts and identify untrusted text. |
| J3 | Track multi-turn risk and apply stricter handling when required. |
| J4 | Moderate generated output. |
| J5 | Return a fixed refusal when refusal is required. |
| A1–A3 | Apply applicable action authorization, taint-aware rules, and memory-write protections. |
| T1, T2, T4, T6, T7 | Apply tool registration, argument validation, confirmation, call limits, and data-flow rules. |
| D1–D4 | Apply retrieval permissions, sensitive-data checks, canary detection, and outbound-destination restrictions. |
| Q-Gate | Detect suspicious untrusted text using a quantum-kernel classifier. |
| RBF baseline | Provide a classical SVM comparison for injection detection. |
| H1 | Generate answers grounded in vetted passages. |
| H2 | Validate passage citations and supported source values. |
| H3 | Optionally evaluate whether claims are entailed by cited passages. |
| H5 | Decide whether to return, prune, or abstain from an answer. |
| M1–M3 | Record events and support audit-chain verification. |

Layer availability depends on the active configuration. Not every optional or recommended component is necessarily enabled in every execution.

**Implementation note:** The current documented pipeline implements H1, H2, H3, and H5. H4 is not included as an implemented layer in the current main-branch pipeline.

## Grounding and Output Verification

Aegis treats answer verification as a separate security stage rather than relying exclusively on the LLM to follow its instructions.

### H1 — Grounded Generation

The model receives vetted passages and is instructed to use them as evidence. It should cite the relevant passage identifiers or indicate that the requested information was not found.

### H2 — Citation and Source-Value Checks

The citation checker verifies that:

- Referenced passage identifiers exist.
- Citations follow the expected format.
- Extracted factual values, particularly numbers, can be found in the cited source text.

H2 is deterministic, but it is not a complete semantic understanding system. Passing its checks does not prove that every statement is factually correct.

### H3 — Natural-Language Inference

H3 uses an NLI model to evaluate whether a claim is entailed by the cited evidence.

The implementation distinguishes entailment, contradiction, and neutral outcomes. It applies a configured confidence threshold before accepting a claim as supported.

NLI availability depends on model initialization and execution. If the model is unavailable, the pipeline may fall back to H2-only verification.

### H5 — Final Output Decision

H5 determines how the answer should be handled:

- **ANSWER:** Return the answer when the required checks pass.
- **PRUNED:** Remove unsupported sentences when supported content remains.
- **ABSTAIN:** Avoid returning an unsupported answer when no acceptable content remains.

For example, if a source states that delivery takes four days but the answer claims nine days, the unsupported claim should not be treated as verified evidence.

## Q-Gate: Quantum-Kernel Injection Detection

Q-Gate is an advisory detector designed to identify possible prompt injection in untrusted text.

### Processing Pipeline

1. Text is transformed into numerical features using TF-IDF character n-grams and dimensionality reduction.
2. The features are scaled for a small quantum feature map.
3. PennyLane's `default.qubit` simulator encodes the features into quantum states.
4. A quantum kernel calculates state similarity.
5. An SVM uses the kernel to classify suspicious text.
6. A classical RBF SVM provides a comparison using the corresponding feature pipeline.

The fidelity kernel is based on the overlap between encoded quantum states:

```text
k(a, b) = |<phi(a) | phi(b)>|²
```

The detector can classify text for passing, review, or quarantine. These results are used by the surrounding security workflow; Q-Gate is not itself the final authorization authority for tool actions.

### Quantum-Security Scope

- Q-Gate runs on a simulator, not real quantum hardware.
- The use of a quantum kernel does not establish quantum advantage.
- Detector performance must be assessed using the corresponding evaluation data and result files.
- A detector verdict alone does not establish whether an attack achieved its objective.

## Simulated Tools

The demonstration includes the following tools:

| Tool | Purpose |
|---|---|
| `search_docs` | Search available demonstration documents. |
| `read_file` | Read permitted demonstration files. |
| `query_db` | Query the simulated database. |
| `write_note` | Write a note in the demonstration environment. |
| `send_email` | Simulate an email action. |

The simulated `send_email` tool writes to a local outbox instead of sending a real email.

These tools are intended for controlled testing and do not represent authorization to access real external systems.

## Audit Logging

Aegis records events in an append-only, hash-chained JSONL audit log.

Each record includes fields such as:

- Sequence number.
- Timestamp.
- Previous record hash.
- Event data.
- Current record hash.

The verifier checks the record sequence, previous-hash links, and record hashes.

This mechanism helps detect modifications, reordering, and deletions within a chain. It does not prove that the original events were truthful or that every security decision was correct. Detecting truncation at the end of a log also requires an independently preserved expected final hash.

### Verify the Audit Log

From the repository root, activate the project environment and run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/verify_audit.py results/audit.jsonl
```

Use the audit file produced by the execution mode being inspected. The policy evaluator and the pipeline evaluator may write to different log files.

A valid audit chain confirms structural integrity according to the verifier; it is not proof of overall security effectiveness.

## Evaluation and Testing

Aegis includes automated tests and evaluation scripts for inspecting individual controls and comparing system configurations.

### Evaluation Measures

Depending on the evaluation mode and available result files, the project can report:

- **Attack Success Rate (ASR):** The proportion of evaluated attacks that achieve their defined objective.
- **Detection rate:** The proportion of relevant attacks detected by the evaluated defense.
- **Benign task success:** The proportion of benign requests completed successfully.
- **Accuracy:** The proportion of evaluated cases classified or handled according to their expected outcomes.
- **Latency:** Time taken to process requests.
- **Q-Gate versus RBF:** A comparison of the quantum-kernel detector and its classical baseline.

Results must be interpreted alongside the evaluation mode, dataset composition, sample size, and configuration. An evaluation containing no attack cases cannot establish attack-prevention effectiveness.

The repository's policy-mode evaluator is a keyword-policy prototype, not a measurement of the complete LLM pipeline. Pipeline-mode evaluation runs the actual agent pipeline and has different runtime and dependency requirements. Their results should not be combined as though they measured the same system.

### Run the Tests

```powershell
python -m pytest -v
```

### Run an End-to-End Check

If the script is present in the checkout:

```powershell
python -m scripts.check_e2e
```

### Run a Pipeline Evaluation

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m eval.run --configs 0_baseline 7_full --workers 1
```

For a quick smoke test, add `--limit 8`. A small smoke test is useful for checking that the pipeline runs, but it is not sufficient for drawing broad security conclusions.

**Important:** Evaluation commands may overwrite result files such as `results/summary.csv` and `results/per_case.csv`, or produce audit logs. Preserve any existing results you need before running a new evaluation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main implementation language |
| Chainlit | Chat interface |
| LiteLLM | LLM provider integration |
| Pydantic | Data contracts and validation |
| PennyLane | Quantum-kernel simulation |
| scikit-learn | SVMs, feature processing, and evaluation |
| PyYAML | Configuration handling |
| SQLite | Simulated database |
| `hashlib` and `hmac` | Hash-chain verification and action-token protection |
| pandas | Evaluation data processing |
| pytest | Automated testing |

Exact dependencies and supported Python versions are defined by the repository's dependency and configuration files.

## Installation and Setup

Run the following commands from the repository root.

### 1. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 2. Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an approved environment-specific workaround or activate the environment from Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a Python version compatible with the dependencies listed in `requirements.txt`.

### 4. Configure Environment Variables

If the repository includes an `.env.example` file, copy it to `.env` and configure the required provider credentials and model settings.

```powershell
Copy-Item .env.example .env
```

Do not commit `.env`, API keys, access tokens, or other secrets.

### 5. Launch the Application

Use the entry point documented by the current checkout. For a Chainlit application with `app.py`, the command is:

```powershell
chainlit run app.py
```

Open the local URL printed by the application.

If an entry point or setup script is absent, check the current repository structure rather than assuming that an optional script exists.

## Example Attack Scenario

Consider a user asking:

> What changed in the vendor update?

A retrieved document contains an instruction attempting to redirect the agent to send sensitive information to an external recipient.

Aegis handles the scenario through several checks:

1. The agent retrieves the relevant document.
2. Ingress identifies untrusted content and applies data-protection checks.
3. Q-Gate evaluates the suspicious text and may flag or quarantine it.
4. The LLM proposes an answer or a tool action.
5. The tool-safety layer evaluates the proposed action against registration, argument, taint, and outbound-destination rules.
6. If the action is denied, the simulated email is not sent through that blocked action.
7. The grounded-response stage generates an answer from vetted evidence.
8. Citation and output-verification checks inspect the answer.
9. Audit events record the relevant decisions for later verification.

This illustrates the intended defense-in-depth design. The outcome of a particular run must be verified using its actual configuration, observable effects, and audit records.

## Limitations

- Q-Gate runs on a quantum simulator; quantum advantage is not claimed.
- Demo tools are simulated and do not replace real-world security controls.
- Input classifiers and injection detectors can miss malicious content or misclassify benign content.
- Heuristic normalization and taint tracking cannot guarantee detection of every obfuscated or transformed payload.
- H2 checks citation structure and source values but does not establish full semantic correctness.
- H3 depends on the availability and reliability of the NLI model.
- A valid audit chain does not prove that its events are true.
- Optional security layers may not be enabled in every configuration.
- Performance claims should be supported by the corresponding result files, test-set sizes, and evaluation configuration.
- A successful test run demonstrates the tested behavior only; it does not prove that every possible attack is prevented.

## Project Principle

**Probabilistic detectors identify suspicious content. Deterministic policy layers control actions. Grounding checks verify answers. The audit chain records the decisions.**