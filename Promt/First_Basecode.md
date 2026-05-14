Dưới đây là **prompt hoàn chỉnh bằng tiếng Anh** để bạn đưa cho Claude Opus 4.6. Tôi đã viết theo hướng: **tự lên kế hoạch + mặc định bắt tay code luôn + được quyền tải mọi thứ cần thiết + liên tục tự kiểm tra và sửa**, đồng thời buộc nó **bám sát paper XG-NID và các nguồn được trích trong paper**, chứ không được tự bịa. XG-NID mô tả pipeline gồm 6 khối: Flow and Feature Generator, Explainable Feature Extractor, Graph Generator, HGNN model, Integrated Gradient Explainer, và Generative Explainer. Paper cũng nói GNN4ID là tool open-source cho 3 khối đầu, flow giới hạn 20 packets, idle timeout 120s, dùng 76 flow-level features, 14 packet-level features, packet payload được biểu diễn thành vector 1500 chiều, heterogeneous graph có flow nodes và packet nodes với “contain” và “link” edges, HGNN dùng 2 lớp GATConv, global mean pooling, rồi fully connected layers; phần giải thích dùng Integrated Gradients và Llama 3-8B theo prompt structure ở Algorithm 3.    

Bạn có thể copy nguyên khối dưới đây:

```text
You are Claude Opus 4.6 acting as a highly autonomous research engineer and reproduction lead.

Your mission is to PLAN, IMPLEMENT, VERIFY, DEBUG, and ITERATE until you produce the most faithful possible codebase reproduction of the paper:

“XG-NID: Dual-Modality Network Intrusion Detection using a Heterogeneous Graph Neural Network and Large Language Model”

Important context:
- I already have access to the paper.
- The authors appear to have publicly released GNN4ID, but not the full end-to-end XG-NID code.
- Therefore, you must infer the missing implementation details carefully from the paper and its cited sources, but you must NOT invent unsupported claims and must clearly label assumptions.
- You are explicitly authorized to download, install, clone, inspect, and use any dependency, repository, paper, model, tool, or dataset needed to complete the reproduction.
- You have broad autonomy. Do not wait for permission. Default to doing the work.
- During the build, you must continuously re-check your work, run tests, compare against the paper, and fix issues whenever needed.
- Accuracy matters more than speed.

Core operating principles:
1. Default to action. Do not stop at planning; begin implementation immediately after the initial plan.
2. Continuously self-audit. After every meaningful step, verify whether the implementation still matches the paper and whether outputs are plausible.
3. Use the paper as the primary specification.
4. Use cited references from the paper when needed to fill in implementation gaps.
5. Use public source code from GNN4ID and any cited/open implementations as secondary evidence.
6. Never silently guess. If a detail is not specified, infer the most defensible implementation, document the assumption, and keep going.
7. Prefer a runnable, testable, modular reproduction over a vague “best effort.”
8. Maintain a persistent engineering log of what you tried, what failed, what you changed, and why.
9. If exact reproduction is impossible because the paper omits critical details, produce the closest paper-faithful implementation and explicitly mark every divergence.
10. Keep iterating until the system runs end-to-end or until you can prove exactly what is blocked and why.

Primary objective:
Build a faithful, runnable XG-NID codebase, including as much of the following as possible:
- raw traffic / PCAP ingestion
- flow generation
- flow feature extraction
- temporal explainable feature extraction
- packet payload extraction and transformation
- heterogeneous graph construction
- HGNN model for graph-level classification
- integrated gradients explainer
- generative explainer using an LLM
- training pipeline
- evaluation pipeline
- inference pipeline
- reproducibility scripts
- configuration files
- documentation
- ablation-ready structure where practical

Paper-faithful requirements to honor unless evidence strongly supports a better interpretation:
- The system has six major components:
  1) Flow and Feature Generator
  2) Explainable Feature Extractor
  3) Graph Generator
  4) GNN Model
  5) Integrated Gradient Explainer
  6) Generative Explainer
- Flow generation should reflect the paper’s real-time intent, including:
  - maximum 20 packets per flow
  - idle timeout of 120 seconds
- Build upon NFStream if appropriate and supported by evidence.
- Compute the described flow-level and packet-level features as faithfully as possible.
- Packet payload representation should be transformed into a 1500-dimensional feature space from bytes, zero-padded when shorter, zero-filled if no payload.
- Heterogeneous graph should include:
  - flow nodes
  - packet nodes
  - “contain” edges from flow to packet nodes
  - “link” edges between sequential packet nodes
- Edge features should reflect paper descriptions:
  - contain edges: layer sizes and packet direction
  - link edges: time difference between consecutive packets
- HGNN should reflect the paper structure as closely as possible:
  - graph attention based message passing
  - 2 GATConv layers
  - normalization / activation as described
  - global mean pooling
  - fully connected classifier head
- Explainability:
  - integrated gradients on the HGNN
  - generative explainer driven by feature importance and payload-specific logic
  - prompt structure should follow the paper’s Algorithm 3 as closely as possible
- Use a modern open LLM or accessible API path for the generative explainer, but if the exact model in the paper is unavailable, choose the closest viable alternative and document the substitution.

Mandatory workflow:
Phase 1 — Specification extraction
- Read the paper carefully and extract an implementation specification.
- Convert the paper into a structured engineering checklist.
- Identify all underspecified details, ambiguities, and hidden dependencies.
- Read the cited references that are necessary to implement missing parts correctly.
- Inspect GNN4ID and determine exactly which parts of the pipeline it already covers.
- Produce a “faithfulness map”:
  - directly specified by paper
  - inferred from cited reference
  - inferred from open-source code
  - engineering assumption

Phase 2 — Environment and dependency setup
- Create a reproducible environment.
- Install all dependencies.
- Clone and inspect relevant repositories.
- Pin versions whenever possible.
- If any dependency breaks, resolve it instead of stopping.
- Prefer clean project structure and reproducible commands.

Phase 3 — Code implementation
- Implement the project in modular form.
- Reuse reliable public code only when appropriate and legal.
- Wrap external code in clean interfaces instead of producing a tangled prototype.
- If GNN4ID covers the first three components, integrate it cleanly and verify it matches the paper.
- Implement missing parts yourself where needed.

Phase 4 — Verification and iterative debugging
- After each component, run sanity checks.
- Validate intermediate artifacts:
  - parsed packets
  - flows
  - feature dimensions
  - graph shapes
  - edge counts
  - class mappings
  - train/val/test splits
  - model outputs
  - explanation outputs
- If any result looks inconsistent with the paper, investigate and correct it.
- Repeat until stable.

Phase 5 — Reproduction evaluation
- Try to reproduce the paper’s reported evaluation setting as closely as possible.
- If exact dataset filtering, balancing, or splits are unclear, implement the most paper-faithful approximation and document the gap.
- Compare obtained results against the paper and explain discrepancies honestly.
- Produce an evaluation summary with:
  - what matched
  - what partially matched
  - what could not be matched
  - likely reasons

Phase 6 — Final deliverables
Produce all of the following:
1. A runnable repository
2. Clear README with setup and usage
3. A “paper-to-code mapping” document
4. A “reproduction gaps and assumptions” document
5. Training and evaluation scripts
6. Example inference and explanation outputs
7. A build log / lab notebook
8. A final summary of fidelity and limitations

Behavioral rules:
- Do not ask me for permission to proceed. Proceed.
- Do not stop after giving a plan. Start coding.
- Do not say “this would require access” unless you have already tried to obtain that access.
- If something fails, debug it and continue.
- If a repository is missing, search for mirrors, forks, archived copies, or alternative implementations.
- If an exact dependency version is unavailable, choose the closest defensible one and document it.
- If multiple plausible implementations exist, choose the one best supported by the paper and cited sources.
- Use concise status updates focused on progress, blockers, fixes, and next actions.

Source discipline:
- Treat the paper as the canonical specification.
- Consult the cited references when the paper leaves implementation details unclear.
- Consult GNN4ID source code directly for the graph/data pipeline.
- Record which source justified each important implementation decision.
- When you infer something, explicitly mark it as an inference.
- Never represent an inference as if it were explicitly stated in the paper.

Dataset and preprocessing discipline:
- Reconstruct the CIC-IoT2023-related pipeline as faithfully as possible.
- Respect the paper’s discussion of attacker MAC filtering and class balancing if reproducing those experiments.
- Preserve subclass proportionality where the paper implies it.
- Build configurable preprocessing so alternate interpretations can be tested.

Engineering quality requirements:
- Use a clean folder structure.
- Add type hints where practical.
- Add assertions and validation checks.
- Add logging.
- Add unit tests or lightweight verification scripts for core data transformations.
- Make configuration values externalized rather than hard-coded where reasonable.
- Avoid hidden magic.

Continuous self-check loop:
At all times, follow this cycle:
1. Read source evidence
2. Implement
3. Run
4. Inspect outputs
5. Compare with paper
6. Identify mismatch
7. Fix
8. Re-run
Do this repeatedly without waiting for instruction.

Default output format during execution:
For each major step, report briefly:
- Goal
- What you did
- Evidence/source used
- Result
- Mismatch or uncertainty
- Next action

Definition of done:
You are not done when you have a plan.
You are not done when code merely exists.
You are done only when you have:
- a runnable end-to-end implementation or the closest provable approximation,
- documented all deviations,
- verified outputs at each stage,
- and provided a clear fidelity assessment against the paper.

Start now.
First:
1. extract the paper into an implementation spec,
2. inspect GNN4ID and all necessary cited sources,
3. create the project scaffold,
4. then begin implementing immediately.
```

Một vài lưu ý để prompt này hiệu quả hơn:

* Prompt trên đã ép Claude phải bám vào các chi tiết paper nêu rõ, ví dụ:

  * **20 packets per flow** và **idle timeout 120s** trong Flow and Feature Generator. 
  * **76 flow-level features**, **14 packet-level features**, và **payload 1500-dim**. 
  * Graph gồm **flow nodes**, **packet nodes**, **contain edges**, **link edges**. 
  * HGNN dùng **2 GATConv layers**, rồi **GlobalMeanPool**, rồi FC head. 
  * Explainability dùng **Integrated Gradients** và **Algorithm 3** cho generative explainer. 
* Paper cũng nói **GNN4ID** là open-source tool cho việc chuyển raw traffic thành graph, nên prompt đã ép Claude phải inspect nó trước thay vì viết bừa.  
* Tôi cũng cố tình nhét vào prompt yêu cầu Claude phải tạo **faithfulness map** và **reproduction gaps**, vì paper không public full code nên sẽ có chỗ phải suy luận. Điều này giúp nó không “ảo tưởng đã reproduce exact”.

Nếu muốn, tôi có thể viết tiếp cho bạn một **bản prompt thứ hai** còn “gắt” hơn, chuyên để ép Claude:

* ưu tiên **PyTorch Geometric**,
* target **Linux + CUDA**,
* tự benchmark,
* và tạo luôn **TODO list theo từng file code**.
