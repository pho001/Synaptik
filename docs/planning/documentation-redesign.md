# Didactic documentation redesign

## Status and authorization

Status: Complete — the user-authorized editorial scope is integrated and qualified below.
The reader-oriented structure, whole-framework explanation, every-package atlas and commented
examples are the execution scope. Existing normative authority and numerical/ownership contracts
do not change. Classification: A; no production Java, native implementation or dependency change.

Depends on: the completed source-matched audit and private `0.1.0-public-audit.20261104` artifacts.
Common base revision: `f3f7c2face52c1fa886e075a68faa6d11765cbd0`.
Parallel group: `DOC-REDESIGN-2026-11-04`, with isolated writer worktrees and exclusive scopes
recorded below. Integration owner: coordinator for navigation, framework explanation, glossary,
link migration outside reference, release-source inventory and combined verification.
Integration order: learning/reference and atlas slices in any order, then shared navigation.
Integration validation: external public-API examples, package/native coverage, local links/anchors,
affected Javadoc review and independent reader-oriented review. Record actual human observations
separately; an automated review is not a human usability study.

## Goal

Make Synaptik usable by Java developers without prior framework knowledge while retaining a
complete, technically accurate path to its implementation. Separate tutorials, task-oriented
how-to guides, conceptual explanation and precise reference. Do not organize the beginner path
around repository modules or expose internal qualification history as introductory material.

## Required deliverables

- Clear entry paths for application developers, ML/tensor users and framework contributors.
- One coherent CPU/FLOAT32 learning project: first model, data/shapes, composition, loss/gradients,
  train/evaluate, save/load, reusable execution, then backend/performance choices.
- Complete commented runnable examples. Each explains its goal, prerequisites, public API usage,
  why important steps exist, expected output and interpretation, one useful modification and
  relevant errors. Comments explain decisions and invariants, not obvious syntax.
- A connected explanation of the whole framework: Tensor/Model/Engine, capture/compile/planning,
  prepare/run/materialize, gradients/training, resources/lifetimes and persistence/reconstruction.
  Follow the same actual program through intermediate artifacts and ownership boundaries.
- An architectural atlas covering every production Java package, including internal packages,
  and corresponding native components. Each describes purpose/non-goals, place in the lifecycle,
  actual mechanism, inputs/outputs, important entry points, callers/callees, dependency direction,
  owned state/resources, errors/invariants and its part in a concrete example. Detail follows
  complexity; a package list or repeated class names is not an explanation.
- Illustrative accessible diagrams for lifecycle, dependencies, memory/ownership, gradients,
  tensor/layout/broadcast geometry, backend transfers and sequence activity where useful.
- Topic-sized reference pages linked to generated Javadoc; concise glossary definitions with
  contextual links. Preserve one authoritative home per contract and avoid copied explanations.
- Separate contributor/maintenance/planning/history navigation from the user learning journey.
  Preserve normative contracts and migrate incoming links rather than silently breaking them.
- Version-specific capability/installation/distribution statements. No unsupported backend or
  public redistribution claim; no promise that saving a model saves optimizer/executable state.

## Acceptance

A new Java reader can navigate, run and modify the first model, train it, save it and load it
without assistance. Validate both an ML-experienced reader and a reader learning the required ML
concepts. Every production package/native component is accounted for in the atlas. Every runnable
example is exercised against the selected real public artifacts; claimed output and ownership
behavior match execution. Canonical example sources and displayed excerpts cannot silently drift.
Record human observations separately from automated link/API checks. Do not add incidental
source-text/default/count tests or rewrite product semantics to make an example appear to work.

## Completed execution slices

The coordinator inventoried the audited selected sources: 78 Java packages, including the three
testing marker packages, plus the Metal native source/build components. The teaching project is
one FLOAT32 linear model with input `[2,1]`, supplied weight zero, no bias, inputs `[1,3]`, targets
`[2,6]`, sum-squared loss and `Sgd(0.05)`. Its first step has loss 40 and weight gradient -40,
yielding weight 2. These fixed numbers connect the tutorial, architectural explanation and atlas.

| Completed brief | Exclusive write scope | Prerequisite |
|---|---|---|
| [DOC0001 learning journey](documentation/tasks/0001-learning-journey.md) | `docs/tutorials/`, `docs/examples/` | Completed audit |
| [DOC0002 Model atlas](documentation/tasks/0002-model-atlas.md) | `docs/explanation/packages/model.md` | Completed audit and package inventory |
| [DOC0003 execution atlas](documentation/tasks/0003-execution-atlas.md) | `docs/explanation/packages/execution.md` | Completed audit and package inventory |
| [DOC0004 CPU atlas](documentation/tasks/0004-cpu-atlas.md) | `docs/explanation/packages/cpu.md` | Completed audit and package inventory |
| [DOC0005 Metal/native atlas](documentation/tasks/0005-metal-atlas.md) | `docs/explanation/packages/metal.md` | Completed audit and native inventory |
| [DOC0006 extensions/tools atlas](documentation/tasks/0006-extensions-atlas.md) | `docs/explanation/packages/extensions-tools.md` | Completed audit and package inventory |
| [DOC0007 reference migration](documentation/tasks/0007-reference-migration.md) | `docs/api/`, `docs/reference/`, `docs/history/reference/` | Completed audit and existing anchor inventory |
| [DOC0008 independent reader/source review](documentation/tasks/0008-reader-review.md) | Read-only; coordinator integrates corrections | Integrated DOC0001–DOC0007 |

All seven editorial slices and the final independent review are complete. The coordinator
integrated shared navigation, framework explanations, glossary, current link migration and
the reviewed source inventory. No documentation frontier remains active in this group.

## Qualification and limits

See [the complete qualification record](documentation-redesign-validation.md): all seven
independent learning modes and the 38-line complete transcript, 55 reference classes / 53 main
programs, 78 packages / 43 native components, 542 preserved incoming anchors, supplied/example
Javadoc, actual local reader navigation and 11 accessible diagrams in light/dark.

Independent automated reader/source reviews found and resolved stale semantic statements.
There were no human-reader observations; this is not a usability study. Framework/authority
inputs and sealed private artifacts remain unchanged. The generated local receipt separately
binds final source selection, source-only packaging and exact verification artifacts; this work
does not qualify a new backend or authorize public distribution.
