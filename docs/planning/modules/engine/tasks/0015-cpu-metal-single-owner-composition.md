# Task 0015: CPU/Metal Single-Owner Engine Composition

## Status

Complete

Frontier verification: the user authorized one sequential, single-writer implementation worktree at
exact base `6f5be27c`. Engine 0014 and Metal 0004 are Complete, their architecture contracts are
stable, and no later Engine task is authorized. Task 0015 therefore owns this complete vertical
slice and its shared explanatory/planning-document reconciliation.

## Change class

Class C — this task changes public construction API, Engine-to-backend dependency exposure,
entry-time ownership and cleanup, native Metal lifecycle, cold Prepare routing, host ingress and
materialization, and the mixed-owner fail-closed boundary. Independent targeted review is
mandatory before final commit.

## Goal

Implement the explicit CPU/Metal Engine builder selected by ADR 0015 as one complete single-owner
vertical slice from registration through real Metal execution and cleanup.

## Scope

- Add public `Engine.builder()` and nested single-use closeable `Engine.Builder` with concrete CPU
  and Metal ownership overloads.
- Freeze one aligned backend identity, provider, and availability snapshot per registration;
  compile from the ordered inventory; reject duplicates without replacing the first owner.
- Route non-empty single-owner plans to the exact registered adapter before analysis and retain that
  adapter directly through prepared/run ingress and materialization.
- Add Metal-owned public configuration and integration, explicit native open/rollback, availability,
  Prepare contribution, caller-host upload, canonical download, and cleanup.
- Keep tuning CPU-only while permitting a CPU-owned plan when Metal is registered; reject a
  Metal-owned plan before borrow or trial work.
- Preserve `Engine.standard()` through the builder, the CPU-only advanced surface, Runtime's direct
  prepared-reference hot path, and reverse outer cleanup.
- Add focused lifecycle/API/architecture coverage and a real opt-in public Metal lifecycle test.

## Non-goals

- No mixed-owner transfer or schedule assembly, automatic fallback, retry, owner substitution,
  backend discovery, plugin SPI, global registry, hot reload, CUDA composition, async execution,
  Metal capability expansion, native ABI change, or Metal tuning exposure.
- No change to public compile, prepare, run, handle, publication, or result signatures beyond the
  explicit construction boundary.

## Contracts

- [`runtime-prepare-engine.md` headings `modules/engine`, `Public explicit composition`, `Prepare
  lifecycle`, and `Runtime service locator`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  own builder transfer, fixed inventory, routing, direct adapter capture, close order, and hot-path
  lookup prohibitions.
- [`backend-execution.md` heading `Metal backend`](../../../../architecture/contracts/backend-execution.md#metal-backend)
  owns Metal configuration, native context, storage, materialization, and physical lifecycle.
- [ADR 0015](../../../../design/decisions/0015-explicit-engine-backend-composition.md) fixes the API
  shape, single-owner first slice, CPU-only tuning boundary, and required evidence.

## Dependencies and integration

- Depends on: Engine 0014; Metal 0004; current Compiler, Prepare 0006, and Runtime 0016 contracts
- Conflicts with: concurrent writes to Engine composition/lifecycle, Metal public/native lifecycle,
  affected shared explanations, or Engine/Metal master plans
- Parallel group: None
- Common base revision: `6f5be27c`
- Integration order: atomic single-writer vertical slice
- Integration validation: affected Engine/Metal/architecture/integration suites, real Metal smoke,
  generated Javadoc, documentation validation, then repository gate by the integration owner
- Shared-document integration owner: this isolated task worktree

## Files and symbols

- `modules/engine` — `Engine.Builder`, ordered registry, CPU/Metal adapters, direct selected-adapter
  prepared/result flow, standard construction, and CPU tuning gate.
- `backends/metal` — `MetalBackendConfiguration`, `MetalBackendIntegration`, package-private native
  lifecycle implementation, generic schedule assembly, host ingress/materialization, and retained
  borrow lifecycle.
- `testing/{architecture-tests,integration-tests}` — dependency/API locks and real public lifecycle.
- Engine/Metal package Javadocs plus public API, backend-selection, Metal backend, lifecycle,
  module-boundary, task, and master-plan documentation.

## Acceptance criteria

- The public nested builder exposes only concrete CPU and Metal ownership overloads; non-null
  ownership transfers at method entry and failed registration closes the candidate with cleanup
  failures suppressed on the primary failure.
- Registration first captures one immutable fact and then mutates one ordered registration
  collection exactly once. Interrogation or insertion failure leaves prior registrations usable,
  rejects equal backend IDs without replacing the first owner, and closes the candidate once.
  Build requires one entry, is single-use and terminal on failure, and builder/Engine cleanup is
  reverse, attempt-all, idempotent, and failure-retaining where specified.
- Compile sees all captured providers/snapshots in registration order. Missing, empty, or mixed
  owner plans fail before any backend preparation; a valid plan calls exactly one adapter.
- Prepared/run/result state carries the direct selected adapter; ingress and materialization do not
  re-query the registry; Runtime performs no registry, provider, configuration, or reflection lookup.
- Metal configuration requires an explicit absolute native-library path. Opening owns ABI/context
  validation and reverses every handle-to-wrapper and context/runtime/integration publication
  boundary on failure with the original primary, distinct-only suppression, and exact-once
  releases. A supported real NEG compiles, prepares, runs, materializes canonical bytes, reuses the
  prepared execution, and releases all native ownership.
- CPU tuning works with Metal registered. Metal-owned tuning fails before representative borrow,
  candidate generation, or trial work. Mixed CPU/Metal preparation fails before backend analysis.
- `Engine.standard()` remains behaviorally CPU-only through the builder; `AdvancedEngine` remains
  CPU-only; Metal production has no Engine dependency. Metal's public package exposes only the
  capability provider, configuration, and integration; no second public native opener exists.
  Engine publishes CPU/Metal construction types as API-visible dependencies.
- Public Javadoc and focused explanatory/API/backend/user/planning documentation describe current
  behavior without converting an explanatory page into authority.

## Validation

Worker/final integration validation:

```bash
./native/metal-macos-arm64/build.sh
./gradlew :modules:engine:test :backends:metal:test
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest'
./gradlew :testing:architecture-tests:test
./gradlew :modules:engine:javadoc :backends:metal:javadoc
./gradlew test
python3 <temporary Markdown link/anchor/heading/fence validator> <changed Markdown paths>
git diff --check
```

The implementation context records exact outcomes below. The independent review reuses successful
Java evidence unless it changes executable behavior.

## Documentation and review impact

General, API/Javadoc, Backend Guide, User Guide, Architecture, and Planning profiles apply. The
existing glossary term `Backend composition` keeps the semantics selected by 0014 but is updated
from planned to current implementation status. Independent Class C review must inspect production
behavior, ownership/failure paths, real integration evidence, public/current wording, link/anchor
validity, dependency direction, and scope against both incorporated contracts before the final
commit.

## Result

Implementation is complete and independently approved:

- Added the public concrete builder, fixed ordered registry, direct selected-adapter lifecycle,
  reverse ownership cleanup, CPU-only tuning gate, and standard-factory cutover.
- Added Metal-owned configuration/integration, native context contribution, host upload, canonical
  download, generic complete-partition schedule assembly, and retained caller-storage borrow.
- Added focused builder/routing/configuration/public-shape coverage and a real public integration
  that reuses one Metal preparation, materializes exact bits, rejects a mixed plan, tunes CPU with
  Metal registered, and rejects Metal tuning before borrow.
- The first independent Class C review returned five P1 findings. The implementation now stores
  each Builder registration as one immutable fact in one collection, injects deterministic
  registration/build faults, and proves candidate exact-once plus reverse accepted-owner rollback
  with primary/suppressed/self-suppression assertions.
- Every Metal native-owner publication edge now has an injected failure seam and reverse
  attempt-all rollback evidence. Metal implementation types moved into the public facade's package
  as package-private types; the former `.internal.MetalBackendRuntime` public opener no longer
  exists, and a reflection shape lock covers the three intended public types.
- The second independent Class C review passed code, lifecycle, Metal ownership, public shape,
  direct-adapter routing, and gate behavior, but returned one P1 repository-wide current-status
  documentation finding. That finding drove the final reconciliation below.
- The first narrow documentation re-review then found unqualified completion-era status statements
  in completed Metal tasks 0001–0003. Those historical acceptance and evidence records now retain
  their original scope in explicit completion-time language and point to task 0015's superseding
  public composition status without rewriting what the earlier tasks delivered.
- The final narrow documentation re-review approved the completed reconciliation. It confirmed
  that no unqualified historical status residual remains and that completion-time wording
  preserves the evidence of Metal tasks 0001–0003.
- Public Javadocs, repository and documentation indexes, ADR 0015, architecture status/dependency
  and Runtime/Prepare boundary explanations, capability and partition-preparer guides, public API
  status, and live planning summaries now state that explicit CPU/Metal registration and
  single-owner Metal execution are current. Only standard-Metal convenience, mixed-owner
  execution, generic plugins, and broader Metal support remain planned.
- `./native/metal-macos-arm64/build.sh` completed successfully.
- The focused Engine/Metal fault suites reran all tasks and completed `BUILD SUCCESSFUL`; affected
  Engine, Metal, and architecture suites and the environment-enabled full repository test gate
  also completed `BUILD SUCCESSFUL`.
- The opt-in real Metal lifecycle and standard CPU composition smoke each completed
  `BUILD SUCCESSFUL`; affected Javadocs regenerated without warnings after the final source
  wording correction.
- Repository-wide validation checked 8,039 local links across 497 Markdown files with no missing
  target or anchor and found no unbalanced fence or heading-level jump. Targeted and
  repository-wide stale-status searches were clean, and `git diff --check` passed.

Independent Class C review is approved: the code/lifecycle re-review passed, and the final narrow
documentation re-review confirmed the repository-wide status and historical evidence.
