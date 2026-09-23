# Task 0014: Explicit Backend Composition Architecture

## Status

Complete

Frontier verification: clean `main` was at exact revision
`83f28005fbfcb6518af3fbd8760297c90ea12e1e`; Engine is Complete through 0013, no later Engine
task existed or was `Ready`, and the user explicitly authorized this first contract-first
workstream. Therefore 0014 is the next Engine master ID and its dependencies are satisfied.
Execution used isolated worktree `/tmp/synaptik-engine-0014` on
`work/engine-0014-composition`.

## Change class

Class C — this task selects a public composition API, cross-module dependency direction,
construction and prepared-resource ownership, native configuration ownership, Prepare routing,
and the mixed-owner fail-closed boundary. It changes architecture and documentation only, with an
independent targeted review; it changes no Java production behavior.

## Goal

Make one complete architecture decision for explicit CPU/Metal Engine composition so later
implementation cannot invent API shape, ownership timing, discovery, routing, rollback, closure,
or mixed-owner semantics.

## Scope

- Select `Engine.builder()` and public `AutoCloseable Engine.Builder` with concrete CPU and Metal
  `takeOwnership(...)` overloads plus `build()` and `close()`.
- Define entry-time ownership transfer, equal-`BackendId` duplicate rejection, registration-time
  capability/availability capture, build rollback, reverse close, and suppressed failures.
- Define Engine's private ordered registry, compile-time owner selection, cold Prepare routing,
  direct selected-adapter ingress/materialization, CPU-only `prepareTuned(...)` semantics,
  single-owner first implementation, and pre-analysis mixed-owner rejection.
- Keep Metal configuration/native open in Metal, preserve `Engine.standard()` as fresh CPU
  convenience, keep backends independent of Engine, and prohibit Runtime/hot-path lookup.
- Synchronize the authoritative contracts, ADR, focused architecture/API/backend explanations,
  glossary term, this task, and Engine master plan.

## Non-goals

- No Java, Javadoc, Gradle, dependency, test-source, native, ABI, capability, numerical, or runtime
  behavior change.
- No implementation scaffold, generic backend SPI, plugin discovery, hot reload, global registry,
  automatic fallback, mixed-owner transfer, CUDA composition, or roadmap edit.
- No claim that the selected builder, Metal configuration/integration, or Metal Engine lifecycle is
  currently callable.

## Contracts

- [`ARCHITECTURE.md` headings `Core lifecycle` and `Core invariants`](../../../../../ARCHITECTURE.md#core-lifecycle)
  — preserve Engine as composition root, prepared/result ownership, backend independence, and
  Runtime hot-path separation.
- [`runtime-prepare-engine.md` headings `modules/engine`, `Prepare lifecycle`, `Runtime service
  locator`, and `Reflective backend plugin discovery`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — owns explicit composition, routing, lifecycle, and discovery prohibitions.
- [`backend-execution.md` heading `Metal backend`](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — owns Metal configuration, native integration, and physical resource policy.

The root authority index already routes Engine composition and Metal implementation to those exact
scoped headings, so `ARCHITECTURE.md` needs no scope-index change.

## Dependencies and integration

- Depends on: Engine 0013; Metal 0004; current Compiler ownership, Prepare 0006, and Runtime 0016
  contracts
- Conflicts with: concurrent edits to Engine composition, lifecycle, ownership, the two scoped
  contracts, or the changed explanatory documents
- Parallel group: `contract-first-83f2800`
- Common base revision: `83f28005fbfcb6518af3fbd8760297c90ea12e1e`
- Integration order: before any Engine explicit-composition or public Metal lifecycle implementation
- Integration validation: `./gradlew :testing:architecture-tests:test` after ordered branch
  integration, then the repository gate selected by the integration owner
- Shared-document integration owner: post-branch shared-document integration owner; exclusively
  owns `docs/planning/roadmap.md`

## Files and symbols

- `docs/architecture/contracts/runtime-prepare-engine.md` — normative Engine builder, registry,
  ownership, routing, closure, and hot-path boundary.
- `docs/architecture/contracts/backend-execution.md` — normative Metal configuration/native owner.
- `docs/design/decisions/0015-explicit-engine-backend-composition.md` — alternatives, decision,
  consequences, and test obligations; the design index and ADR 0006 receive navigation links.
- `docs/architecture/{lifecycle,module-boundaries,dependency-rules,runtime-prepare-backend-boundary}.md`
  — focused lifecycle and dependency explanations.
- `docs/api/public-api.md`, `docs/backend-guide/metal-backend.md`, and `docs/glossary.md` — planned
  public shape, Metal author guidance, and shared term.
- this task and `docs/planning/modules/engine/master-plan.md` — authorization, result, row, and
  frontier.

## Acceptance criteria

- The planned public type/method shape and current-versus-planned status are unambiguous.
- Ownership has one owner at every registration/build/Engine stage; failures perform reverse,
  attempt-all cleanup with the primary throwable preserved and distinct failures suppressed.
- Equal duplicate IDs fail without replacing the first entry; provider, snapshot, and integration
  IDs agree; availability is frozen once per Engine construction.
- Compile selects owner IDs from the registration-ordered frozen inventory; cold Prepare routes by
  owner and captures its adapter directly in outward handles; Runtime performs no
  registry/adapter/provider/configuration lookup.
- The first slice permits one distinct registered owner and rejects missing, empty, or mixed owner
  sets before backend analysis, with no retry or fallback.
- `prepareTuned(...)` accepts a single CPU-owned plan even with Metal registered, rejects Metal
  before trials, and keeps safe-heuristic fallback within the selected CPU owner.
- Metal owns configuration/native open; `Engine.standard()` remains CPU-only; no backend depends on
  Engine; no Java production behavior changes.
- The ADR, focused explanations, glossary, task, and master are synchronized; the root index and
  global roadmap remain unchanged for stated reasons.

## Architecture-test plan

The existing `EngineCompositionContractTest` remains the executable guard that every production
backend is independent of Engine. No prospective source-text test is added before the API exists.
The implementation task must:

1. extend `EngineTypedPublicShapeTest` for `Engine.builder`, `Engine.Builder`, both concrete
   ownership overloads, `build`, and `close`;
2. add focused lifecycle cases for entry-time transfer, duplicate IDs, snapshot-once timing,
   rollback, reverse closure, suppression order, handle-before-backend closure, and spent builder;
3. add CPU-only and Metal-only Prepare routing plus pre-analysis missing/empty/mixed-owner rejection;
4. prove direct selected-adapter caller ingress and result materialization without a registry
   lookup;
5. cover CPU tuning with Metal registered, pre-trial Metal rejection, and CPU-only fallback;
6. update `EngineCompositionContractTest` for the intentional API-visible Engine-to-CPU/Metal
   edges while retaining the forbidden backend-to-Engine direction; and
7. retain Runtime hot-path structural validation proving no registry or adapter lookup.

## Validation

Worker validation:

```bash
./gradlew :testing:architecture-tests:test
python3 <temporary Markdown link/anchor/heading/fence validator> <changed Markdown paths>
git diff --check
git diff --name-only 83f28005fbfcb6518af3fbd8760297c90ea12e1e --
```

Also verify exact changed-path scope, final newlines, no trailing whitespace, unique headings,
balanced fences, local links/anchors, the task size guardrail, no Java/build/native changes, no
roadmap change, and a targeted glossary match. Repository-wide validation is deferred to the
integration owner because this branch changes no executable behavior or build graph.

## Documentation and review impact

- General, Architecture, API/Javadoc, Backend Guide, Decision Record, and Planning profiles apply.
- Javadoc impact: none; no Java type or behavior changes.
- Glossary impact: add `Backend composition` because the selected reusable term now has exact
  construction, registry, and fail-closed meaning.
- Independent Class C review is mandatory and must inspect the final diff against the two scoped
  contracts, current source/tests, authority boundaries, planned/current labeling, links, and test
  plan without rerunning already successful Java validation.

## Result

Complete:

- Selected the public builder and concrete ownership-overload shape, entry-time transfer,
  duplicate-ID and fixed-snapshot rules, transactional construction, reverse closure, and
  suppressed-failure semantics.
- Fixed compile/Prepare routing at construction/cold time, direct selected-adapter
  ingress/materialization, and CPU-only tuning/fallback semantics; selected the single-owner first
  slice and kept mixed-owner execution fail-closed pending an explicit transfer contract.
- Kept Metal configuration/native open in Metal, `Engine.standard()` CPU-only, backend dependency
  direction one-way, and Runtime free of registry or adapter lookup.
- Updated two scoped contracts, ADR 0015, focused lifecycle/module/API/backend explanations,
  dependency guidance, glossary, and Engine planning; the root index and roadmap correctly remain
  unchanged.
- `./gradlew :testing:architecture-tests:test` completed `BUILD SUCCESSFUL`; targeted validation
  passed all 14 changed Markdown files, the exact 14-path scope and `git diff --check` were clean,
  and independent final review approved with no blockers.
- No Java, Javadoc, build, native, or production behavior changed.
- Unresolved blockers: none. Follow-up is the separately authorized implementation task and, later,
  a distinct mixed-owner transfer decision before that boundary can be relaxed.

Status: Complete
