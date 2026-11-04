# DOC0007: Topic-sized API reference and anchor migration

Status: Complete. Classification: A; integrated editorial work and bounded qualification.

## Authorization and coordination

User continued the queued didactic redesign after the completed public-readiness audit. Coordinator verified this authorized frontier, source inventory and disjoint file/semantic ownership before launch.

- Depends on: completed public-readiness audit.
- Conflicts with: no other writer within the exclusive scope below; never change shared product contracts.
- Parallel group: DOC-REDESIGN-2026-11-04.
- Common base revision: `f3f7c2face52c1fa886e075a68faa6d11765cbd0`.
- Integration order: Any; coordinator integrates learning/reference, then atlas and shared navigation.
- Integration validation: coordinator compiles/runs canonical examples against installed private artifacts, checks package/native coverage and local links/anchors, and runs independent reader-oriented/source review. No worker build/lint/tests/formatters mid-flight.
- Shared-document integration owner: coordinator.

## Scope

Exclusive writes: `docs/api/*.md; docs/reference/**; docs/history/reference/**`. Move existing large reference material into focused topics, retaining useful technical detail and all incoming heading/explicit anchors through old-path navigation. Reconcile current source-backed stage/capability/numerical statements; separate superseded qualification history rather than overriding contradictions with a generic banner. Preserve all incoming anchors and publish any changed mappings. The coordinator owns final verification/extraction wording in `docs/reference/javadoc.md`; do not edit it during this correction pass. No other current/historical files are in this slice.

Read `AGENTS.md`, documentation general style and the matching user-guide/architecture/API profile, example-format where relevant. Root `ARCHITECTURE.md` scope index and exact relevant headings:

- ARCHITECTURE.md: Authority, incorporation, and precedence
- docs/architecture/contracts/foundational-modules.md: Module responsibilities
- docs/architecture/contracts/compiler-autograd.md: Compile artifacts
- docs/architecture/contracts/runtime-prepare-engine.md: modules/engine
- docs/architecture/contracts/extensions-training.md: Owned tensor Models; Generic graph source and values

## Shared interfaces

The complete source/package/native inventory and fixed learning arithmetic are in `local://documentation-redesign-inventory.json`. Canonical entry points are `docs/tutorials/index.md`, `docs/examples/README.md`, `docs/explanation/framework.md`, `docs/explanation/packages/index.md`, and `docs/reference/index.md`; coordinator owns the explanation/navigation except the learning/reference slices. Keep documentation English, matching existing repository conventions. All source examples use real supported public API and the audited private `0.1.0-public-audit.20261104` artifacts. No public licence/publishing authorization; no numerical, API, ownership or dependency changes.

## Acceptance and handoff

Teach purpose, actual mechanism and why, inputs/outputs, entry points/callers/dependencies, owned state/resources, failures/invariants and the package's part (or absence) in the canonical learning example. Do not substitute class inventories, generic boilerplate or inferred mechanisms for source reading. Link important symbols to their actual source/Javadoc and precise existing authority; internal packages remain unsupported application API. Include accessible diagrams only where they teach relationships. No duplicated normative authority or copying qualification history into a beginner path. Every runnable example has prerequisites, concrete inputs, rationale, expected output/interpretation, modification and relevant errors. Do not claim a program was run until coordinator executes it.

Use the assigned isolated worktree, commit only exclusive files and return commit ID, paths, source/contract evidence, unexecuted verification commands and concrete caveats. Skip build/lint/tests/formatters; coordinator runs the integrated checks once afterward. Report `Status: Complete` only for your implemented editorial handoff with deferred runtime proof explicitly identified.

## Completion and evidence

Initial topic migration integrated in `e50ad6daf`; source-fidelity/archive corrections integrated in `23e5e0d6f`. All 542 incoming original anchors survive, current caller migration has zero remaining old API links, all 55 complete reference classes compile, all 53 mains pass, and all 47 literal stdout blocks match.

See [the shared qualification record](../../documentation-redesign-validation.md) for exact scope,
generated receipts, source-freeze boundaries and the distinction between automated review and
human observations. Production contracts, licensing and the sealed audit candidate are unchanged.
