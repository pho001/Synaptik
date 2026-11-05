# Autotuning candidate coverage and correctness audit

Status: Complete

## Authorization and ordering

The user requested this audit after the completed [documentation comprehension pass](../../../documentation-terminology-validation.md), before live CLI work and matched PyTorch measurements. The user explicitly forbids implementing architecture-changing repairs: analyze those gaps and propose a solution instead.

- **Depends on:** completed documentation comprehension pass; delivered Tuning 0001–0005 and current Engine/backend producers.
- **Conflicts with:** edits to tuning transactions, Engine tuning orchestration or CPU/Metal candidate producers outside this task.
- **Parallel group:** read-only `autotuning-audit`; implementation scopes assigned only after findings are grounded.
- **Common base revision:** `7ec6de7865ac456ac52108e84165286ac375a146`.
- **Integration order:** generic transaction and producer audits first; bounded authorized repairs and independent review next; coordinator runs executable verification once after integration.
- **Integration validation:** affected tuning/Engine/backend tests plus actual public Engine tuning scenarios and exact numerical results; no test-only assertion is evidence that native execution occurred.
- **Shared-document integration owner:** main coordinator.
- **Frontier verification:** user authorization is current; documentation prerequisite and source archive acceptance completed; no other source writer active; audit scopes below are read-only and semantically separate.

## Contracts

Read [AGENTS.md](../../../../../AGENTS.md). The authoritative root is [ARCHITECTURE.md](../../../../../ARCHITECTURE.md): “Core invariants” and “Dependency rules”. Applicable scoped authority:

- [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules) and [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- [Runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime), [Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare), and [Compile ownership and mixed-owner Prepare routing](../../../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing).

Generic tooling measures opaque backend-supplied candidates; it does not invent backend settings, select ownership or change arithmetic. The current Engine contract explicitly admits only an eligible single-owner CPU plan or singleton-NEG Metal plan and forbids heterogeneous tuning trials. Exhaustive backend-combination search therefore requires an architecture proposal, not silently removing this guard. Runtime performs no search or cache access.

## Independent audit scopes

1. **Generic transactions:** `tools/tuning` production and focused tests. Check candidate validation, all-candidate execution, correctness-before-timing, fresh mutable inputs and cleanup obligations, budgets/overflow, medians/ties, cache keys/decoding/corruption/atomicity and session versus persistent reuse. Report exact consumer-visible defects and executable reproduction scenarios; distinguish contract requirements from framework inspiration.
2. **Producer and Engine coverage:** Engine tuning orchestration and CPU/Metal candidate generators plus their focused tests and explicit configuration. Enumerate actual tunable families, route/configuration variants, pruning, limits, oneDNN/OpenBLAS/vector/scalar coverage, backend combinations and unsupported domains. Trace omitted variants to source and authority. Separate implementation bugs from architecture-dependent expansion.

The coordinator compares primary Triton, PyTorch and TVM materials, integrates findings, executes real scenarios, implements only repairs that preserve current ownership/lifecycle/dependency/numerical contracts, and commissions the required independent review. Audit workers do not build, test, lint, format or edit files.

## Acceptance

- An evidence-backed coverage matrix distinguishes available execution routes, actual tuning candidates, supported combinations, explicit exclusions and search limits. No “all variants” claim without a defined finite domain.
- Real runs prove eligible CPU and Metal transactions, their candidate counts and numerical results, cache behavior and fail-before-trial rejection of forbidden combinations; record hardware, exact libraries and command boundaries.
- Reproduce concrete defects before repair; retain behavior-focused regressions where justified and rerun the changed real scenario afterward.
- Architecture-changing gaps receive a concrete owner-by-owner proposal, identity/compatibility impact, correctness/resource requirements and tradeoffs; do not implement that expansion.
- Update affected Javadoc and explanatory documentation without rewriting historical evidence or requalifying sealed runtime artifacts. Final status records actual exercised verification and unresolved prerequisites, not inferred success.

## Grounded repair frontiers

Both read-only audits are complete. Actual unchanged-reference runs reproduced both clock rollover
failures, primary-error masking in both cache publishers, 596 incoherent snapshots among 10,000
reads during valid atomic workload-cache replacement, and strict public CPU MATMUL rejection.
The safe CPU fallback did execute the same MATMUL and returned exactly `[19,22,43,50]`.
Six actual Metal requests measured both declared local candidates and the sole complete plan,
preserved signed-zero results and performed no cache-path I/O.

Two clean implementation scopes are Ready at the common base above:

- **Transaction repair:** only `tools/tuning` production/focused tests. Correct modular elapsed
  subtraction, read the opened file's size rather than a separate pathname version, and preserve
  the primary persistence failure when temporary cleanup also fails. Retain all limits, cache
  formats, callback ownership and atomic replacement. Do not promise concurrent writer merging.
- **CPU coverage repair:** CPU-owned producers/composition and Engine's existing tuning adapter,
  plus their focused/integration tests. Repair truthful caller-input representation eligibility,
  Phase-1 decision freezing, the unreachable no-local-workload complete-plan path and the
  artificial rejection of a complete singleton plan. Enumerate already-implemented legal CPU
  route/configuration alternatives, including qualified oneDNN choices, only within existing
  owner, resource, numerical and dependency contracts. Version affected identities and migrate
  callers without compatibility shims. If any expansion needs architecture changes, analyze
  that expansion instead; do not disguise it as a local fix.

Write scopes and semantic ownership are disjoint. Integrate transaction repair before CPU
coverage repair. Workers do not run checks; the coordinator owns the combined executable gate,
failing-before/passing-after smoke and required independent targeted review. Shared documentation
remains coordinator-owned; implementation Javadoc belongs to its writer.

## Result

The [audit report](../../../autotuning-audit.md#verified-cpu-coverage-and-fixes) records the
bounded domain, actual before/after repairs, finite formulas, native policies, cache reuse,
optional-Vector execution and the important observed exact-floating-agreement failure.
Mixed ownership, wider Metal qualification, cross-topology local mapping and a newly tolerant
comparison contract are proposals only, as the user required.

Coordinator verification: 85 Tuning tests and 412 affected CPU/Engine/integration/conformance/
architecture tests, all with zero failures, errors or skips; actual native and public-consumer
programs are recorded separately. An earlier unfiltered CPU timeout is not called a pass.
Independent source reviewers' findings were repaired; final follow-up reviews found no remaining
source defect. The copied current lifecycle Java example also executes outside the checkout,
without the optional Vector module, and prints the four expected MATMUL values.

No Model arithmetic, owner/lifetime, Runtime search, module-edge or native ABI change is part
of this cutover. Historical sealed documentation/release qualification remains unchanged.
