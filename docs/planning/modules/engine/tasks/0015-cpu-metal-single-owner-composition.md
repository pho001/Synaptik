# Task 0015: [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) Single-[Owner](../../../../glossary.md#term-owner) [Engine](../../../../glossary.md#term-engine) Composition

> **Editorial historical reading key.**
>
> This task implements the preceding composition decision as a single-owner vertical slice. Its
> mixed-owner rejection and pre-trial Metal tuning rejection describe that delivery, not today's full
> surface; [current
> composition](../../../../reference/engine/composition.md#current-explicit-cpu-and-metal-composition)
> documents later extensions. Builder receives caller-opened integrations at method entry, records
> aligned identity/provider/availability facts, and transfers accepted ownership to Engine only on
> successful build. Registration and native-wrapper publication failures retain their own rollback
> responsibilities.
>
> Compiler selects semantic owners from the captured inventory; Engine routes cold preparation and
> retains direct backend adapters. Metal supplies native context ownership, host upload, canonical
> download, and physical cleanup. Runtime executes prepared references rather than consulting the
> registry. The [current ordinary
> lifecycle](../../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
> distinguishes borrowed caller storage, result-owned completed resources, and explicit detached
> copies. The recorded review iterations and execution evidence remain completion-time history.

## Status

Complete

Frontier verification: the user authorized one sequential, single-writer implementation worktree at
exact base `6f5be27c`. [Engine](../../../../glossary.md#term-engine) 0014 and [Metal](../../../../glossary.md#term-metal) 0004 are Complete, their architecture contracts are
stable, and no later Engine task is authorized. Task 0015 therefore owns this complete vertical
[slice](../../../../glossary.md#slice) and its shared explanatory/[planning](../../../../glossary.md#term-planning)-document reconciliation.

## Change class

Class C — this task changes public construction [API](../../../../glossary.md#term-application-programming-interface-api), [Engine](../../../../glossary.md#term-engine)-to-[backend](../../../../glossary.md#backend) [dependency](../../../../glossary.md#term-dependency) exposure,
entry-time [ownership](../../../../glossary.md#term-ownership-meanings) and cleanup, native [Metal](../../../../glossary.md#term-metal) [lifecycle](../../../../glossary.md#lifecycle), cold [Prepare](../../../../glossary.md#prepare) routing, [host](../../../../glossary.md#term-host) ingress and
[materialization](../../../../glossary.md#materialization), and the mixed-[owner](../../../../glossary.md#term-owner) [fail-closed](../../../../glossary.md#term-fail-closed) boundary. Independent targeted review is
mandatory before final [commit](../../../../glossary.md#term-commit).

## Goal

Implement the explicit [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) [Engine](../../../../glossary.md#term-engine) builder selected by [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0015 as one complete single-[owner](../../../../glossary.md#term-owner)
vertical [slice](../../../../glossary.md#slice) from registration through real Metal execution and cleanup.

## [Scope](../../../../glossary.md#term-scope)

- Add public `Engine.builder()` and nested single-use closeable `Engine.Builder` with concrete CPU
  and Metal ownership overloads.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Freeze one aligned backend identity, provider, and availability snapshot per registration;
  compile from the ordered inventory; reject duplicates without replacing the first owner.<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid); [snapshot glossary](../../../../glossary.md#term-snapshot); [compile glossary](../../../../glossary.md#compile); [owner glossary](../../../../glossary.md#term-owner))</span>
- Route non-empty single-owner plans to the exact registered adapter before analysis and retain that
  adapter directly through prepared/run ingress and materialization.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [materialization glossary](../../../../glossary.md#materialization))</span>
- Add Metal-owned public configuration and integration, explicit native open/rollback, availability,
  Prepare contribution, caller-host upload, canonical download, and cleanup.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [Prepare glossary](../../../../glossary.md#prepare); [host glossary](../../../../glossary.md#term-host); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Keep tuning CPU-only while permitting a CPU-owned plan when Metal is registered; reject a
  Metal-owned plan before borrow or trial work.
- Preserve `Engine.standard()` through the builder, the CPU-only advanced surface, Runtime's direct
  prepared-reference hot path, and reverse outer cleanup.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Add focused [lifecycle](../../../../glossary.md#lifecycle)/[API](../../../../glossary.md#term-application-programming-interface-api)/architecture coverage and a real opt-in public [Metal](../../../../glossary.md#term-metal) lifecycle test.

## Non-goals

- No mixed-owner transfer or schedule assembly, automatic fallback, retry, owner substitution,
  backend discovery, plugin SPI, global registry, hot reload, CUDA composition, async execution,
  Metal capability expansion, native ABI change, or Metal tuning exposure.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [backend glossary](../../../../glossary.md#backend); [plugin glossary](../../../../glossary.md#term-plugin); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [registry glossary](../../../../glossary.md#term-registry); [CUDA glossary](../../../../glossary.md#term-cuda); [Metal glossary](../../../../glossary.md#term-metal); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- No change to public compile, prepare, run, handle, publication, or result signatures beyond the
  explicit construction boundary.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [signatures glossary](../../../../glossary.md#term-signature))</span>

## Contracts

- [`runtime-prepare-engine.md` headings `modules/engine`, `Public explicit composition`, `Prepare
  lifecycle`, and `Runtime service locator`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  own builder transfer, fixed inventory, routing, direct adapter capture, close order, and hot-path
  lookup prohibitions.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [`backend-execution.md` heading `Metal backend`](../../../../architecture/contracts/backend-execution.md#metal-backend)
  owns Metal configuration, native context, storage, materialization, and physical lifecycle.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [materialization glossary](../../../../glossary.md#materialization); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR 0015](../../../../design/decisions/0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span> fixes the API
  shape, single-owner first slice, CPU-only tuning boundary, and required evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [owner glossary](../../../../glossary.md#term-owner); [slice glossary](../../../../glossary.md#slice); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Dependencies and integration

- Depends on: [Engine](../../../../glossary.md#term-engine) 0014; [Metal](../../../../glossary.md#term-metal) 0004; current [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare) 0006, and [Runtime](../../../../glossary.md#term-runtime) 0016 contracts
- Conflicts with: concurrent writes to Engine composition/lifecycle, Metal public/native lifecycle,
  affected shared explanations, or Engine/Metal master plans<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None
- Common base revision: `6f5be27c`
- Integration order: [atomic](../../../../glossary.md#resource-transaction-and-rollback) single-writer vertical [slice](../../../../glossary.md#slice)
- Integration validation: affected Engine/Metal/architecture/integration suites, real Metal smoke,
  generated Javadoc, documentation validation, then repository gate by the integration owner<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [smoke glossary](../../../../glossary.md#term-smoke-test); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): this isolated task worktree

## Files and symbols

- `modules/engine` — `Engine.Builder`, ordered registry, CPU/Metal adapters, direct selected-adapter
  prepared/result flow, standard construction, and CPU tuning gate.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal))</span>
- `backends/metal` — `MetalBackendConfiguration`, `MetalBackendIntegration`, package-private native
  lifecycle implementation, generic schedule assembly, host ingress/materialization, and retained
  borrow lifecycle.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- `testing/{architecture-tests,integration-tests}` — [dependency](../../../../glossary.md#term-dependency)/[API](../../../../glossary.md#term-application-programming-interface-api) locks and real public [lifecycle](../../../../glossary.md#lifecycle).
- Engine/Metal package Javadocs plus public API, backend-selection, Metal backend, lifecycle,
  module-boundary, task, and master-plan documentation.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- The public nested builder exposes only concrete CPU and Metal ownership overloads; non-null
  ownership transfers at method entry and failed registration closes the candidate with cleanup
  failures suppressed on the primary failure.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Registration first captures one immutable fact and then mutates one ordered registration
  collection exactly once. Interrogation or insertion failure leaves prior registrations usable,
  rejects equal backend IDs without replacing the first owner, and closes the candidate once.
  Build requires one entry, is single-use and terminal on failure, and builder/Engine cleanup is
  reverse, attempt-all, idempotent, and failure-retaining where specified.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner); [Engine glossary](../../../../glossary.md#term-engine); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- Compile sees all captured providers/snapshots in registration order. Missing, empty, or mixed
  owner plans fail before any backend preparation; a valid plan calls exactly one adapter.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Prepared/run/result state carries the direct selected adapter; ingress and materialization do not
  re-query the registry; Runtime performs no registry, provider, configuration, or reflection lookup.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [materialization glossary](../../../../glossary.md#materialization); [registry glossary](../../../../glossary.md#term-registry); [Runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Metal configuration requires an explicit absolute native-library path. Opening owns ABI/context
  validation and reverses every handle-to-wrapper and context/runtime/integration publication
  boundary on failure with the original primary, distinct-only suppression, and exact-once
  releases. A supported real NEG compiles, prepares, runs, materializes canonical bytes, reuses the
  prepared execution, and releases all native ownership.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- CPU tuning works with Metal registered. Metal-owned tuning fails before representative borrow,
  candidate generation, or trial work. Mixed CPU/Metal preparation fails before backend analysis.
- `Engine.standard()` remains behaviorally [CPU](../../../../glossary.md#term-cpu)-only through the builder; `AdvancedEngine` remains
  CPU-only; Metal production has no Engine dependency. Metal's public package exposes only the
  capability provider, configuration, and integration; no second public native opener exists.
  Engine publishes CPU/Metal construction types as API-visible dependencies.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Public Javadoc and focused explanatory/API/backend/user/planning documentation describe current
  behavior without converting an explanatory page into authority.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## [Validation](../../../../glossary.md#term-validation)

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

General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, User Guide, Architecture, and [Planning](../../../../glossary.md#term-planning) profiles apply. The
existing glossary term [`Backend composition`](../../../../glossary.md#backend-composition) keeps the semantics selected by 0014 but is updated
from planned to current implementation status. Independent Class C review must inspect production
behavior, [ownership](../../../../glossary.md#term-ownership-meanings)/failure paths, real integration evidence, public/current wording, link/anchor
validity, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), and [scope](../../../../glossary.md#term-scope) against both incorporated contracts before the final
[commit](../../../../glossary.md#term-commit).

## Result

Implementation is complete and independently approved:

- Added the public concrete builder, fixed ordered registry, direct selected-adapter lifecycle,
  reverse ownership cleanup, CPU-only tuning gate, and standard-factory cutover.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Added Metal-owned configuration/integration, native context contribution, host upload, canonical
  download, generic complete-partition schedule assembly, and retained caller-storage borrow.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [canonical glossary](../../../../glossary.md#term-canonical-form); [partition glossary](../../../../glossary.md#partition))</span>
- Added focused builder/routing/configuration/public-shape coverage and a real public integration
  that reuses one Metal preparation, materializes exact bits, rejects a mixed plan, tunes CPU with
  Metal registered, and rejects Metal tuning before borrow.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- The first independent Class C review returned five P1 findings. The implementation now stores
  each Builder registration as one immutable fact in one collection, injects deterministic
  registration/build faults, and proves candidate exact-once plus reverse accepted-owner rollback
  with primary/suppressed/self-suppression assertions.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [owner glossary](../../../../glossary.md#term-owner); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Every Metal native-owner publication edge now has an injected failure seam and reverse
  attempt-all rollback evidence. Metal implementation types moved into the public facade's package
  as package-private types; the former <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [facade glossary](../../../../glossary.md#term-facade))</span>`.internal.MetalBackendRuntime` public opener no longer
  exists, and a reflection shape lock covers the three intended public types.
- The second independent Class C review passed code, lifecycle, Metal ownership, public shape,
  direct-adapter routing, and gate behavior, but returned one P1 repository-wide current-status
  documentation finding. That finding drove the final reconciliation below.
- The first narrow documentation re-review then found unqualified completion-era status statements
  in completed Metal tasks 0001–0003. Those historical acceptance and evidence records now retain
  their original scope in explicit completion-time language and point to task 0015's superseding
  public composition status without rewriting what the earlier tasks delivered.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- The final narrow documentation re-review approved the completed reconciliation. It confirmed
  that no unqualified historical status residual remains and that completion-time wording
  preserves the evidence of Metal tasks 0001–0003.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>
- Public Javadocs, repository and documentation indexes, ADR 0015, architecture status/dependency
  and Runtime/Prepare boundary explanations, capability and partition-preparer guides, public API
  status, and live planning summaries now state that explicit CPU/Metal registration and
  single-owner Metal execution are current. Only standard-Metal convenience, mixed-owner
  execution, generic plugins, and broader Metal support remain planned.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- `./native/metal-macos-arm64/build.sh` completed successfully.
- The focused [Engine](../../../../glossary.md#term-engine)/[Metal](../../../../glossary.md#term-metal) fault suites reran all tasks and completed `BUILD SUCCESSFUL`; affected
  Engine, Metal, and architecture suites and the environment-enabled full repository test gate
  also completed `BUILD SUCCESSFUL`.
- The opt-in real Metal [lifecycle](../../../../glossary.md#lifecycle) and standard [CPU](../../../../glossary.md#term-cpu) composition [smoke](../../../../glossary.md#term-smoke-test) each completed
  `BUILD SUCCESSFUL`; affected Javadocs regenerated without warnings after the final source
  wording correction.
- Repository-wide validation checked 8,039 local links across 497 Markdown files with no missing
  target or anchor and found no unbalanced fence or heading-level jump. Targeted and
  repository-wide stale-status searches were clean, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.

Independent Class C review is approved: the code/lifecycle re-review passed, and the final narrow
documentation re-review confirmed the repository-wide status and historical evidence.
