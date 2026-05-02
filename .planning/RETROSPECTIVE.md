# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 - Accelerator Runtime Architecture

**Shipped:** 2026-04-30
**Phases:** 5 | **Plans:** 16 | **Tasks:** 41

### What Was Built

- Backend-neutral accelerator buffer layout ABI for Metal now and CUDA later.
- Metal layout-aware device flow using dense physical logical-view buffers and explicit CPU materialization boundaries.
- Materialization-aware accelerator planning with selected/rejected cost summaries and CPU hot-path safeguards.
- Tuning/profile ownership split between graph autotune and platform calibration, with runtime-derived accelerator costs.
- Accelerator observability closure through report contracts, transformer-block workload coverage, Metal/CUDA gates, docs, and source hygiene.

### What Worked

- Verification gates stayed focused and fast by using targeted Gradle filters instead of default benchmark-heavy test runs.
- Keeping `Tensor` logical and moving residency into runtime state preserved the public API while enabling accelerator work.
- Explicit trace/report contracts made fallback, materialization, copy timing, and residency testable rather than advisory.

### What Was Inefficient

- Planning metadata lagged implementation twice: Phase 5 needed a late `05-VERIFICATION.md`, and roadmap/project status needed cleanup before the final audit could pass.
- Local tuning profile files remained dirty throughout the close and required repeated staging discipline.

### Patterns Established

- Backend-neutral accelerator contracts live in common packages; native handles and native ABI assumptions stay backend-owned.
- Legal logical views can use dense physical device buffers while Java owns logical scatter materialization until a versioned native layout ABI exists.
- Benchmark reports are explain artifacts and profile-read-only; autotune/calibration own persistence.
- Milestone closure should always run audit-open, milestone audit, explicit staging checks, and archive generation before deleting living requirements.

### Key Lessons

1. Treat phase-level `VERIFICATION.md` as a hard artifact gate before milestone audit.
2. Keep audit status, project status, roadmap coverage, and state files synchronized immediately after verification.
3. Guard local profile/calibration output with tests, ignore rules, and explicit `git add` paths.
4. Runtime-derived cost inputs should flow through audited `RuntimeConfig`, not local profile file reads.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended close-out run plus phase execution sessions recorded in `STATE.md`.
- Notable: targeted verification and report-contract tests reduced the need for full-suite reruns while still preserving milestone confidence.

---

## Milestone: v1.1 - CUDA Native Runtime

**Shipped:** 2026-04-30
**Phases:** 3 | **Plans:** 10 | **Tasks:** 28

### What Was Built

- Checked-in CUDA graph shim source with optional Gradle build/probe tasks and portable bridge diagnostics.
- Layered CUDA capability diagnostics, shared buffer ABI preflight, and REQUIRED-mode guards for dense `FLOAT32` CUDA paths.
- Native CUDA buffer resources, prepared execution binding, graph-output/CPU-consumer materialization, and adjacent CUDA handoff.
- CUDA trace and benchmark report parity for execution path, reason codes, bytes, copy timing, materialization, and storage residency.
- Developer docs, hygiene gates, portable verification, and native CUDA skip evidence for environments without `nvcc`.

### What Worked

- Capability-gated native checks let the milestone close portably while still documenting real CUDA hardware coverage gaps.
- Fake bridge tests covered CUDA reason-code and required-mode behavior without depending on local GPU availability.
- Keeping CUDA native handles under `backend.cuda.*` preserved the shared accelerator ABI and avoided public `Tensor` API drift.

### What Was Inefficient

- The milestone needed a second audit pass because Phase 6 and Phase 8 Nyquist artifacts lagged behind verification status.
- The earlier audit temporarily treated Phase 8 observability as a gap, so roadmap/state language required cleanup after Phase 8 closure.
- Local tuning profile files stayed dirty and required explicit staging discipline throughout the close.

### Patterns Established

- CUDA buffer support starts narrow: dense `FLOAT32`, same CUDA backend, available native ABI, and supported DAG node types.
- Required CUDA buffer mode throws before tensor-array or CPU fallback can hide native-buffer unavailability.
- CUDA reports use backend-neutral byte/copy/materialization fields, with compatibility fallbacks for older Metal-specific fields.
- Optional native CUDA verification is pass-or-skip by local capability; portable Java gates remain mandatory.

### Key Lessons

1. Keep validation artifacts synchronized with verification immediately, especially when a milestone audit will read both.
2. Preserve narrow runtime support boundaries until native execution, materialization, handoff, and observability are all proven together.
3. Record native-toolchain skip evidence explicitly so portable closure does not masquerade as hardware validation.
4. Continue excluding local tuning/profile output from milestone commits unless intentionally updating canonical fixtures.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended v1.1 run covering Phase 6 through milestone close.
- Notable: portable CUDA fake-bridge coverage reduced dependency on local hardware while preserving reason-code confidence.

---

## Milestone: v1.2 - GPU Region Coverage

**Shipped:** 2026-05-01
**Phases:** 5 | **Plans:** 19 | **Tasks:** 41

### What Was Built

- Native layout ABI v2 metadata, capability/version checks, optional native symbols, and ABI-specific fallback diagnostics for Metal and CUDA.
- GPU layout transform/view paths that preserve legal metadata-only views and route dense materialization through explicit capability gates.
- A backend-neutral Metal/CUDA lowering coverage matrix with stable operation families, support statuses, and unsupported reason codes.
- Safe GPU compound region metadata for `LINEAR_BIAS_ACTIVATION` and `ELEMENTWISE_CHAIN`, with CPU `FUSED` kept explicitly CPU-only.
- Coverage benchmark/report contracts and regression gates for GPU coverage ratio, CPU materialization, hidden tensor-array fallback, and device handoffs.

### What Worked

- Three-source milestone audit caught stale artifacts before archive: requirements, summaries, and verification files all had to agree.
- Capability-gated native checks kept Metal evidence real and CUDA evidence honest without blocking portable closure.
- Treating GPU fusion as compound DAG execution avoided coupling accelerator work to CPU ASM/vector internals.

### What Was Inefficient

- Phase 12 verification existed as summary/validation/security evidence before the phase-level `12-VERIFICATION.md`, so the first milestone audit correctly failed.
- Phase 09 validation frontmatter lagged behind actual test evidence and needed a cleanup pass before archive.
- Local tuning profile files remained dirty throughout the milestone and required repeated staging checks.

### Patterns Established

- Layout ABI v2 support is optional-symbol/version gated and backend-neutral at the metadata layer.
- Metadata-only view propagation and dense GPU materialization are separate runtime paths with distinct reason codes.
- GPU lowering coverage should be represented as a checked-in matrix, not informal backend-specific knowledge.
- Coverage gates should fail on hidden CPU exits and lost GPU region coverage, not only on raw performance thresholds.

### Key Lessons

1. Milestone audit should run after security, validation, and phase verification for every phase, not before artifact cleanup.
2. Phase-level verification is the source of truth for requirement satisfaction even when plan summaries and tests already prove the work.
3. Native CUDA evidence must stay explicit about capability skips until a CUDA-capable host runs the native gate.
4. Longer GPU residency requires coordinated layout, lowering, fusion, and coverage reporting; any one layer alone is insufficient.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended v1.2 run covering Phase 9 through milestone close.
- Notable: targeted Gradle filters plus audit-readable artifacts kept closure fast while still catching real process gaps.

---

## Milestone: v1.3 - Coverage-Driven GPU Region Expansion

**Shipped:** 2026-05-01
**Phases:** 8 | **Plans:** 30 | **Tasks:** 9

### What Was Built

- Coverage gap triage and hot-path targets for transformer, MLP, conv, and normalization-style GPU region work.
- Java-side lowered-region manifest contracts with original op, primitive, dtype/layout, fused subpattern, and rejection metadata.
- BFLOAT16, INT32, and BOOL device residency evidence through runtime binding, backend decisions, traces, and reports.
- Normalization, reduction, softmax-ish, conv, and loss-adjacent lowering coverage or stable rejection detail under a shared Metal/CUDA contract.
- Region-internal GPU fusion for elementwise chains and matmul/linear epilogues without reusing CPU fused ASM internals.
- Multi-op GPU region execution plus hard coverage gates that report lowered counts, fused subpatterns, CPU exits, materialization reasons, and device handoffs.

### What Worked

- The Phase 14 target registry kept later work coverage-driven instead of opportunistic.
- Treating GPU fusion as region-internal lowering/fusion kept CPU `Operation.OpType.FUSED` and generated ASM isolated.
- Hard coverage reports made "hot path stayed on GPU" an audit artifact rather than a timing impression.
- Phase 21 was useful as an evidence-only closure phase: it fixed stale audit proof without touching runtime behavior.

### What Was Inefficient

- Phase 14 and Phase 18 initially lacked phase-level verification reports even though summaries and validation evidence existed.
- Phase 20 validation metadata needed a later hardening pass before the milestone audit could be fully machine-readable.
- Several older summaries did not consistently expose `requirements-completed` frontmatter, so final audit had to cross-check requirement IDs in body text.
- Local tuning profile files stayed dirty and required repeated staging checks.

### Patterns Established

- GPU regions are backend-owned lowered DAGs, not single opaque operations.
- Fusion belongs inside the selected GPU region and is represented as lowering metadata, not CPU fused ASM reuse.
- Unsupported coverage should carry stable rejection detail tied to hot-path targets, dtype/layout legality, and backend capability.
- Milestone audits should include Nyquist coverage for every phase, including evidence-only closure phases.

### Key Lessons

1. Require `VERIFICATION.md` and `VALIDATION.md` for every phase before first milestone audit.
2. Keep `requirements-completed` frontmatter consistent in closure summaries; it makes three-source audits faster.
3. Coverage gates should report structural residency evidence, not only benchmark timing.
4. Evidence-only closure phases should explicitly prove they did not widen runtime or backend behavior.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended v1.3 run covering phases 14 through 21 and archive close.
- Notable: targeted Gradle filters plus grep-based artifact gates kept late evidence repair focused without rerunning unrelated suites.

---

## Milestone: v1.4 - Native GPU Operation Coverage Closure

**Shipped:** 2026-05-02
**Phases:** 7 | **Plans:** 26 | **Tasks:** not recorded

### What Was Built

- Source-of-truth target coverage and semantics contracts for the v1.4 fallback families.
- Native/lowered forward reductions for legal Metal/CUDA dense `FLOAT32` paths.
- GPU-resident LayerNorm and RMSNorm lowering through reduction-adjacent sub-DAGs.
- Verified Metal unmasked forward SDPA admission with CUDA and unsupported SDPA variants kept as stable capability fallbacks.
- Loss/indexing, conv/pool, and BOOL-output support-or-rejection coverage that preserves adjacent GPU region diagnostics.
- Hard coverage regression closure with native evidence, coverage deltas, backend path counters, fallback counts, CPU materialization counts, and local artifact hygiene.

### What Worked

- The target coverage truth table prevented supported rows from becoming paper support without executable backend evidence.
- Keeping unsupported families as explicit capability/rejection evidence allowed the milestone to improve real GPU residency without hiding remaining gaps.
- Final coverage gates focused on structural execution evidence rather than timing-only benchmark claims.
- Late audit normalization made phases 22 through 28 consistently readable by verification, security, validation, and milestone audit workflows.

### What Was Inefficient

- Several early v1.4 phases lacked aggregate verification/security/validation files until the close-out pass.
- The GSD roadmap analyzer did not fully recognize some manually normalized status fields, so the final audit had to rely on disk evidence and explicit archive artifacts.
- Local benchmark/profile outputs remained dirty and required repeated discipline to keep out of milestone evidence.
- CUDA native execution remains locally unproven because `nvcc` is unavailable in this environment.

### Patterns Established

- A coverage row marked supported must map to real backend-lowerable/native execution evidence, not only planner optimism.
- GPU support closure should distinguish native buffer, tensor-array, CPU fallback, candidate shortening, and capability rejection in reports.
- Support-or-rejection milestones can still improve accelerator quality when unsupported cases preserve adjacent GPU regions and stable reason codes.
- Milestone audits should verify security and Nyquist artifacts for every phase before archive, even if implementation tests already passed.

### Key Lessons

1. Mark phase-level security and validation as required close artifacts, not optional after-work.
2. Keep archive copies immutable once a milestone ships; living roadmap and requirements should stay small.
3. CUDA capability skips are acceptable local evidence only when portable fallback and reason-code behavior are tested.
4. Native operation coverage needs both semantic contracts and hard execution-path gates to avoid false positives.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended v1.4 run covering phases 22 through 28 and archive close.
- Notable: coverage regression gates and targeted Gradle filters kept verification focused despite a broad operation-family milestone.

---

## Milestone: v1.5 - Production-Grade Metal Backend Expansion

**Shipped:** 2026-05-02
**Phases:** 11 | **Plans:** 44 | **Tasks:** not recorded

### What Was Built

- Versioned Metal dtype ABI and capability truth across storage, external input, compute, output, operation roles, and reason codes.
- Scoped BF16 compute/output, BOOL-producing compute, and INT32 index/gather/take execution with native Metal evidence or stable rejection.
- GPU-side layout repair, masked/causal SDPA, conv/pool forward execution, dense loss lowering, and selected backward rows.
- Scatter/index-gradient and index-target loss blockers with CPU parity, duplicate/bounds semantics, and visible reason codes.
- Metal route/copy evidence for MPSGraph, custom-kernel seam, tensor-array fallback, CPU fallback, and `MPSGRAPH_RESULT_COPY`.
- Validation, security, verification, coverage gates, and milestone audit artifacts for all v1.5 phases.

### What Worked

- The milestone rule that a Metal row needs semantic contract, lowering, legality, execution, CPU parity, trace/report evidence, and regression coverage prevented overclaiming.
- Backend-neutral contracts still let Metal move first without silently widening CUDA behavior.
- Coverage gates made support-or-rejection work reviewable: unsupported scatter, CUDA parity, custom kernels, and output-buffer writes are visible rather than hidden.
- Closing security and Nyquist artifacts before archive made the final milestone audit a tech-debt review instead of a functional blocker.

### What Was Inefficient

- Security and validation artifacts again lagged implementation and had to be backfilled after the first v1.5 audit.
- `gsd-sdk query milestone.complete` generated low-quality accomplishment text, so MILESTONES/ROADMAP/PROJECT needed manual cleanup.
- Local tuning/profile artifacts remained dirty throughout the close and required explicit staging discipline.
- True MPSGraph output-buffer write proof and real custom Metal kernels remain deferred, so route/copy closure is evidence-based rather than a full zero-copy implementation.

### Patterns Established

- Dtype support must be role-specific: storage/residency is not compute/output support.
- Metal support can be widened safely when every new family has native or routed execution proof plus explicit unsupported variants.
- Copy strategy should be a named report field; `MPSGRAPH_RESULT_COPY` is acceptable only when not mislabeled as zero-copy.
- Phase archival should happen immediately after milestone close to keep `.planning/phases/` ready for the next cycle.

### Key Lessons

1. Run security and Nyquist validation before the first milestone audit, not after audit reports missing artifacts.
2. Treat generated milestone archive text as draft output; inspect and repair user-facing summaries before commit.
3. Do not equate "custom-kernel route exists" with "custom kernels execute"; the seam needs real kernel bridge and parity tests in a later milestone.
4. The next backend milestone should choose between CUDA parity, true output-buffer write proof, and custom Metal kernels based on measured coverage gaps.

### Cost Observations

- Model mix: not recorded for this milestone.
- Sessions: one extended v1.5 run covering phases 29 through 39 and archive close.
- Notable: targeted Gradle filters and coverage gates kept validation feasible despite broad Metal dtype, NN, indexing, loss, training, and router scope.

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | multiple | 5 | Established GSD phase verification, security, Nyquist validation, UAT diagnosis, and milestone audit as required close gates. |
| v1.1 | multiple | 3 | Added capability-gated CUDA native runtime closure with portable fake-bridge verification and explicit native skip evidence. |
| v1.2 | multiple | 5 | Added three-source requirement audit discipline across GPU layout, lowering, fusion, and coverage gates. |
| v1.3 | multiple | 8 | Added coverage-driven GPU region expansion, region-internal fusion, multi-op execution, hard coverage gates, and evidence-only audit closure. |
| v1.4 | multiple | 7 | Added native/lowered GPU operation coverage closure, support-or-rejection diagnostics, and hard execution-path regression gates for high-impact fallback families. |
| v1.5 | multiple | 11 | Expanded Metal toward production-grade coverage with dtype, layout, NN/index/loss/training, router, copy-strategy, security, and validation closure. |

### Cumulative Quality

| Milestone | Requirements | Verification | Open Threats |
|-----------|--------------|--------------|--------------|
| v1.0 | 24/24 satisfied | 5/5 phases passed | 0 |
| v1.1 | 9/9 satisfied | 3/3 phases passed | 0 |
| v1.2 | 16/16 satisfied | 5/5 phases passed | 0 |
| v1.3 | 24/24 satisfied | 8/8 phases passed | 0 |
| v1.4 | 21/21 satisfied | 7/7 phases passed | 0 |
| v1.5 | 33/33 satisfied | 11/11 phases passed | 0 |

### Top Lessons

1. Artifact consistency is part of done; implementation evidence is insufficient if audit-readable files are stale.
2. Accelerator performance work needs observability contracts as much as implementation changes.
3. Native accelerator work should separate portable correctness gates from local hardware/toolchain evidence.
4. GPU residency improvements need layout, lowering, fusion, and coverage gates to evolve together.
5. Evidence-only closure phases are valid when the implementation is complete but audit-readable proof is stale.
6. Native coverage claims should require executable backend evidence or explicit capability-gated rejection, never support labels alone.
7. Milestone archive automation should be treated as a helper, not a substitute for a final human-readable summary pass.
