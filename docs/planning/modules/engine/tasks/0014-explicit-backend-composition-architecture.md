# Task 0014: Explicit [Backend Composition](../../../../glossary.md#backend-composition) Architecture

## Status

Complete

Frontier verification: clean `main` was at exact revision
`83f28005fbfcb6518af3fbd8760297c90ea12e1e`; [Engine](../../../../glossary.md#term-engine) is Complete through 0013, no later Engine
task existed or was [`Ready`](../../../../glossary.md#term-ready), and the user explicitly authorized this first contract-first
workstream. Therefore 0014 is the next Engine master ID and its dependencies are satisfied.
Execution used isolated worktree `/tmp/synaptik-engine-0014` on
`work/engine-0014-composition`.

## Change class

Class C — this task selects a public composition [API](../../../../glossary.md#term-application-programming-interface-api), cross-[module](../../../../glossary.md#term-module) [dependency direction](../../../../glossary.md#java-platform-and-build-tools),
construction and prepared-resource [ownership](../../../../glossary.md#term-ownership), native configuration [ownership](../../../../glossary.md#term-ownership-meanings), [Prepare](../../../../glossary.md#prepare) routing,
and the mixed-[owner](../../../../glossary.md#term-owner) [fail-closed](../../../../glossary.md#term-fail-closed) boundary. It changes architecture and documentation only, with an
independent targeted review; it changes no Java production behavior.

## Goal

Make one complete architecture decision for explicit [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) [Engine](../../../../glossary.md#term-engine) composition so later
implementation cannot invent [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape), [ownership](../../../../glossary.md#term-ownership-meanings) timing, discovery, routing, [rollback](../../../../glossary.md#term-rollback), closure,
or mixed-[owner](../../../../glossary.md#term-owner) semantics.

## [Scope](../../../../glossary.md#term-scope)

- Select `Engine.builder()` and public `AutoCloseable Engine.Builder` with concrete [CPU](../../../../glossary.md#term-cpu) and [Metal](../../../../glossary.md#term-metal)
  `takeOwnership(...)` overloads plus `build()` and `close()`.
- Define entry-time [ownership](../../../../glossary.md#term-ownership-meanings) transfer, equal-[`BackendId`](../../../../glossary.md#backend-identity--backendid) duplicate rejection, registration-time
  capability/availability capture, build rollback, reverse close, and suppressed failures.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Define Engine's private ordered registry, compile-time owner selection, cold Prepare routing,
  direct selected-adapter ingress/materialization, CPU-only <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [registry glossary](../../../../glossary.md#term-registry); [compile glossary](../../../../glossary.md#compile); [owner glossary](../../../../glossary.md#term-owner); [Prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization))</span>`prepareTuned(...)` semantics,
  single-owner first implementation, and pre-analysis mixed-owner rejection.
- Keep Metal configuration/native open in Metal, preserve `Engine.standard()` as fresh CPU
  convenience, keep backends independent of Engine, and prohibit Runtime/hot-path lookup.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Synchronize the authoritative contracts, ADR, focused architecture/API/backend explanations,
  glossary term, this task, and Engine master plan.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Non-goals

- No Java, Javadoc, Gradle, dependency, test-source, native, ABI, capability, numerical, or runtime
  behavior change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- No implementation scaffold, generic backend SPI, plugin discovery, hot reload, global registry,
  automatic fallback, mixed-owner transfer, CUDA composition, or roadmap edit.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [plugin glossary](../../../../glossary.md#term-plugin); [registry glossary](../../../../glossary.md#term-registry); [owner glossary](../../../../glossary.md#term-owner); [CUDA glossary](../../../../glossary.md#term-cuda))</span>
- No claim that the selected builder, Metal configuration/integration, or Metal Engine lifecycle is
  currently callable.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Contracts

- [`ARCHITECTURE.md` headings `Core lifecycle` and `Core invariants`](../../../../../ARCHITECTURE.md#core-lifecycle)
  — preserve Engine as composition root, prepared/result ownership, backend independence, and
  Runtime hot-path separation.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`runtime-prepare-engine.md` headings `modules/engine`, `Prepare lifecycle`, `Runtime service
  locator`, and `Reflective backend plugin discovery`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — owns explicit composition, routing, lifecycle, and discovery prohibitions.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [`backend-execution.md` heading `Metal backend`](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — owns Metal configuration, native integration, and physical resource policy.

The root authority index already routes [Engine](../../../../glossary.md#term-engine) composition and [Metal](../../../../glossary.md#term-metal) implementation to those exact
scoped headings, so `ARCHITECTURE.md` needs no [scope](../../../../glossary.md#term-scope)-index change.

## Dependencies and integration

- Depends on: Engine 0013; Metal 0004; current Compiler ownership, Prepare 0006, and Runtime 0016
  contracts<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Conflicts with: concurrent edits to Engine composition, lifecycle, ownership, the two scoped
  contracts, or the changed explanatory documents
- Parallel group: `contract-first-83f2800`
- Common base revision: `83f28005fbfcb6518af3fbd8760297c90ea12e1e`
- Integration order: before any [Engine](../../../../glossary.md#term-engine) explicit-composition or public [Metal](../../../../glossary.md#term-metal) [lifecycle](../../../../glossary.md#lifecycle) implementation
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew :testing:architecture-tests:test` after ordered branch
  integration, then the repository gate selected by the integration owner<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Shared-document integration owner: post-branch shared-document integration owner; exclusively
  owns `docs/planning/roadmap.md`

## Files and symbols

- `docs/architecture/contracts/runtime-prepare-engine.md` — normative Engine builder, registry,
  ownership, routing, closure, and hot-path boundary.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [registry glossary](../../../../glossary.md#term-registry); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `docs/architecture/contracts/backend-execution.md` — [normative](../../../../glossary.md#term-normative) [Metal](../../../../glossary.md#term-metal) configuration/native [owner](../../../../glossary.md#term-owner).
- `docs/design/decisions/0015-explicit-engine-backend-composition.md` — alternatives, decision,
  consequences, and test obligations; the design index and ADR 0006 receive navigation links.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- `docs/architecture/{lifecycle,module-boundaries,dependency-rules,runtime-prepare-backend-boundary}.md`
  — focused lifecycle and dependency explanations.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- `docs/api/public-api.md`, `docs/backend-guide/metal-backend.md`, and `docs/glossary.md` — planned
  public shape, Metal author guidance, and shared term.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- this task and `docs/planning/modules/engine/master-plan.md` — authorization, result, row, and
  frontier.

## Acceptance criteria

- The planned public type/method [shape](../../../../glossary.md#shape) and current-versus-planned status are unambiguous.
- Ownership has one owner at every registration/build/Engine stage; failures perform reverse,
  attempt-all cleanup with the primary throwable preserved and distinct failures suppressed.<span data-glossary-navigation="true"> ([Ownership glossary](../../../../glossary.md#term-ownership-meanings); [owner glossary](../../../../glossary.md#term-owner); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Equal duplicate IDs fail without replacing the first entry; provider, snapshot, and integration
  IDs agree; availability is frozen once per Engine construction.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Compile selects owner IDs from the registration-ordered frozen inventory; cold Prepare routes by
  owner and captures its adapter directly in outward handles; Runtime performs no
  registry/adapter/provider/configuration lookup.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- The first slice permits one distinct registered owner and rejects missing, empty, or mixed owner
  sets before backend analysis, with no retry or fallback.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [backend glossary](../../../../glossary.md#backend))</span>
- `prepareTuned(...)` accepts a single CPU-owned plan even with Metal registered, rejects Metal
  before trials, and keeps safe-heuristic fallback within the selected CPU owner.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- [Metal](../../../../glossary.md#term-metal) owns configuration/native open; `Engine.standard()` remains CPU-only; no backend depends on
  Engine; no Java production behavior changes.
- The ADR, focused explanations, glossary, task, and master are synchronized; the root index and
  global roadmap remain unchanged for stated reasons.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Architecture-test plan

The existing `EngineCompositionContractTest` remains the executable [guard](../../../../glossary.md#term-guard) that every production
[backend](../../../../glossary.md#backend) is independent of [Engine](../../../../glossary.md#term-engine). No prospective source-text test is added before the [API](../../../../glossary.md#term-application-programming-interface-api) exists.
The implementation task must:

1. extend `EngineTypedPublicShapeTest` for `Engine.builder`, `Engine.Builder`, both concrete
   ownership overloads, <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`build`, and `close`;
2. add focused lifecycle cases for entry-time transfer, duplicate IDs, snapshot-once timing,
   rollback, reverse closure, suppression order, handle-before-backend closure, and spent builder;<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [snapshot glossary](../../../../glossary.md#term-snapshot); [rollback glossary](../../../../glossary.md#term-rollback))</span>
3. add [CPU](../../../../glossary.md#term-cpu)-only and [Metal](../../../../glossary.md#term-metal)-only [Prepare](../../../../glossary.md#prepare) routing plus pre-analysis missing/empty/mixed-[owner](../../../../glossary.md#term-owner) rejection;
4. prove direct selected-adapter caller ingress and result materialization without a registry
   lookup;<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
5. cover CPU tuning with Metal registered, pre-trial Metal rejection, and CPU-only fallback;
6. update `EngineCompositionContractTest` for the intentional API-visible Engine-to-CPU/Metal
   edges while retaining the forbidden backend-to-Engine direction; and
7. retain [Runtime](../../../../glossary.md#term-runtime) hot-path structural [validation](../../../../glossary.md#term-validation) proving no [registry](../../../../glossary.md#term-registry) or adapter [lookup](../../../../glossary.md#term-lookup).

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :testing:architecture-tests:test
python3 <temporary Markdown link/anchor/heading/fence validator> <changed Markdown paths>
git diff --check
git diff --name-only 83f28005fbfcb6518af3fbd8760297c90ea12e1e --
```

Also verify exact changed-path [scope](../../../../glossary.md#term-scope), final newlines, no trailing whitespace, unique headings,
balanced fences, local links/anchors, the task size guardrail, no Java/build/native changes, no
roadmap change, and a targeted glossary match. Repository-wide validation is deferred to the
integration [owner](../../../../glossary.md#term-owner) because this branch changes no executable behavior or build [graph](../../../../glossary.md#graph).

## Documentation and review impact

- General, Architecture, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, Decision Record, and [Planning](../../../../glossary.md#term-planning) profiles apply.
- Javadoc impact: none; no Java type or behavior changes.
- Glossary impact: add [`Backend composition`](../../../../glossary.md#backend-composition) because the selected reusable term now has exact
  construction, registry, and fail-closed meaning.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Independent Class C review is mandatory and must inspect the final diff against the two scoped
  contracts, current source/tests, authority boundaries, planned/current labeling, links, and test
  plan without rerunning already successful Java validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Result

Complete:

- Selected the public builder and concrete ownership-overload shape, entry-time transfer,
  duplicate-ID and fixed-snapshot rules, transactional construction, reverse closure, and
  suppressed-failure semantics.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [shape glossary](../../../../glossary.md#shape); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Fixed compile/Prepare routing at construction/cold time, direct selected-adapter
  ingress/materialization, and CPU-only tuning/fallback semantics; selected the single-owner first
  slice and kept mixed-owner execution fail-closed pending an explicit transfer contract.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [slice glossary](../../../../glossary.md#slice); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Kept [Metal](../../../../glossary.md#term-metal) configuration/native open in Metal, `Engine.standard()` CPU-only, backend dependency
  direction one-way, and Runtime free of registry or adapter lookup.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [Runtime glossary](../../../../glossary.md#term-runtime); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Updated two scoped contracts, ADR 0015, focused lifecycle/module/API/backend explanations,
  dependency guidance, glossary, and Engine planning; the root index and roadmap correctly remain
  unchanged.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [planning glossary](../../../../glossary.md#term-planning))</span>
- `./gradlew :testing:architecture-tests:test` completed `BUILD SUCCESSFUL`; targeted validation
  passed all 14 changed Markdown files, the exact 14-path scope and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` were clean,
  and independent final review approved with no blockers.
- No Java, [Javadoc](../../../../glossary.md#term-javadoc), build, native, or production behavior changed.
- Unresolved blockers: none. Follow-up is the separately authorized implementation task and, later,
  a distinct mixed-owner transfer decision before that boundary can be relaxed.

Status: Complete
