# Didactic documentation redesign

## Status and authorization

Status: Complete — the user-authorized editorial [scope](../glossary.md#term-scope) is integrated and qualified below.
The reader-oriented structure, whole-framework explanation, every-package atlas and commented
examples are the execution scope. Existing [normative](../glossary.md#term-normative) authority and numerical/[ownership](../glossary.md#term-ownership-meanings) contracts
do not change. Classification: A; no production Java, native implementation or [dependency](../glossary.md#term-dependency) change.

Depends on: the completed source-matched audit and private `0.1.0-public-audit.20261104` [artifacts](../glossary.md#term-artifact).
Common base revision: `f3f7c2face52c1fa886e075a68faa6d11765cbd0`.
Parallel group: `DOC-REDESIGN-2026-11-04`, with isolated writer worktrees and exclusive scopes
recorded below. Integration [owner](../glossary.md#term-owner): coordinator for navigation, framework explanation, glossary,
link migration outside reference, [release](../glossary.md#term-release)-source inventory and combined verification.
Integration order: learning/reference and atlas slices in any order, then shared navigation.
Integration [validation](../glossary.md#term-validation): external public-[API](../glossary.md#term-application-programming-interface-api) examples, package/native coverage, local links/anchors,
affected [Javadoc](../glossary.md#term-javadoc) review and independent reader-oriented review. Record actual human observations
separately; an automated review is not a human usability study.

## Goal

Make Synaptik usable by Java developers without prior framework knowledge while retaining a
complete, technically accurate path to its implementation. Separate tutorials, task-oriented
how-to guides, conceptual explanation and precise reference. Do not organize the beginner path
around repository [modules](../glossary.md#term-module) or expose internal [qualification](../glossary.md#term-qualification) history as introductory material.

## Required deliverables

- Clear entry paths for application developers, [ML](../glossary.md#term-machine-learning-ml)/[tensor](../glossary.md#tensor) users and framework contributors.
- One coherent CPU/FLOAT32 learning project: first model, data/shapes, composition, loss/gradients,
  train/evaluate, save/load, reusable execution, then backend/performance choices.<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [FLOAT32 glossary](../glossary.md#term-float32); [model glossary](../glossary.md#term-model); [shapes glossary](../glossary.md#shape); [loss glossary](../glossary.md#loss); [gradients glossary](../glossary.md#term-gradient); [backend glossary](../glossary.md#backend))</span>
- Complete commented runnable examples. Each explains its goal, prerequisites, public API usage,
  why important steps exist, expected output and interpretation, one useful modification and
  relevant errors. Comments explain decisions and invariants, not obvious syntax.<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
- A connected explanation of the whole framework: Tensor/Model/Engine, capture/compile/planning,
  prepare/run/materialize, gradients/training, resources/lifetimes and persistence/reconstruction.
  Follow the same actual program through intermediate artifacts and ownership boundaries.<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine); [capture glossary](../glossary.md#term-capture-meanings); [compile glossary](../glossary.md#compile); [planning glossary](../glossary.md#term-planning); [prepare glossary](../glossary.md#prepare); [run glossary](../glossary.md#term-run); [persistence glossary](../glossary.md#term-persistence); [artifacts glossary](../glossary.md#term-artifact); [ownership glossary](../glossary.md#term-ownership-meanings))</span>
- An architectural atlas covering every production Java package, including internal packages,
  and corresponding native components. Each describes purpose/non-goals, place in the lifecycle,
  actual mechanism, inputs/outputs, important entry points, callers/callees, dependency direction,
  owned state/resources, errors/invariants and its part in a concrete example. Detail follows
  complexity; a package list or repeated class names is not an explanation.<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle); [dependency direction glossary](../glossary.md#java-platform-and-build-tools))</span>
- Illustrative accessible diagrams for lifecycle, dependencies, memory/ownership, gradients,
  tensor/layout/broadcast geometry, backend transfers and sequence activity where useful.<span data-glossary-navigation="true"> ([layout glossary](../glossary.md#layout))</span>
- Topic-sized reference pages linked to generated Javadoc; concise glossary definitions with
  contextual links. Preserve one authoritative home per contract and avoid copied explanations.<span data-glossary-navigation="true"> ([Javadoc glossary](../glossary.md#term-javadoc))</span>
- Separate contributor/maintenance/planning/history navigation from the user learning journey.
  Preserve normative contracts and migrate incoming links rather than silently breaking them.<span data-glossary-navigation="true"> ([normative glossary](../glossary.md#term-normative))</span>
- Version-specific capability/installation/distribution statements. No unsupported backend or
  public redistribution claim; no promise that saving a model saves optimizer/executable state.<span data-glossary-navigation="true"> ([distribution glossary](../glossary.md#term-distribution); [optimizer glossary](../glossary.md#term-optimizer))</span>

## Acceptance

A new Java reader can navigate, [run](../glossary.md#term-run) and modify the first [model](../glossary.md#term-model), train it, save it and load it
without assistance. Validate both an [ML](../glossary.md#term-machine-learning-ml)-experienced reader and a reader learning the required ML
concepts. Every production package/native component is accounted for in the atlas. Every runnable
example is exercised against the selected real public [artifacts](../glossary.md#term-artifact); claimed output and [ownership](../glossary.md#term-ownership-meanings)
behavior match execution. [Canonical](../glossary.md#term-canonical-form) example sources and displayed excerpts cannot silently drift.
Record human observations separately from automated link/[API](../glossary.md#term-application-programming-interface-api) checks. Do not add incidental
source-text/default/count tests or rewrite product semantics to make an example appear to work.

## Completed execution slices

The coordinator inventoried the audited selected sources: 78 Java packages, including the three
testing marker packages, plus the [Metal](../glossary.md#term-metal) native source/build components. The teaching project is
one [FLOAT32](../glossary.md#term-float32) [linear](../glossary.md#term-linear-meanings) [model](../glossary.md#term-model) with input `[2,1]`, supplied [weight](../glossary.md#term-weight) zero, no [bias](../glossary.md#term-bias), inputs `[1,3]`, [targets](../glossary.md#term-target)
`[2,6]`, sum-squared [loss](../glossary.md#loss) and `Sgd(0.05)`. Its first step has loss 40 and weight [gradient](../glossary.md#term-gradient) -40,
yielding weight 2. These fixed numbers connect the tutorial, architectural explanation and atlas.

| Completed brief | Exclusive write [scope](../glossary.md#term-scope) | Prerequisite |
|---|---|---|
| [DOC0001 learning journey](documentation/tasks/0001-learning-journey.md) | `docs/tutorials/`, `docs/examples/` | Completed audit |
| [DOC0002 Model atlas](documentation/tasks/0002-model-atlas.md) | `docs/explanation/packages/model.md` | Completed audit and package inventory |
| [DOC0003 execution atlas](documentation/tasks/0003-execution-atlas.md) | `docs/explanation/packages/execution.md` | Completed audit and package inventory |
| [DOC0004 CPU atlas](documentation/tasks/0004-cpu-atlas.md)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu))</span> | `docs/explanation/packages/cpu.md` | Completed audit and package inventory |
| [DOC0005 Metal/native atlas](documentation/tasks/0005-metal-atlas.md) | `docs/explanation/packages/metal.md` | Completed audit and native inventory |
| [DOC0006 extensions/tools atlas](documentation/tasks/0006-extensions-atlas.md) | `docs/explanation/packages/extensions-tools.md` | Completed audit and package inventory |
| [DOC0007 reference migration](documentation/tasks/0007-reference-migration.md) | `docs/api/`, `docs/reference/`, `docs/history/reference/` | Completed audit and existing anchor inventory |
| [DOC0008 independent reader/source review](documentation/tasks/0008-reader-review.md) | Read-only; coordinator integrates corrections | Integrated DOC0001–DOC0007 |

All seven editorial slices and the final independent review are complete. The coordinator
integrated shared navigation, framework explanations, glossary, current link migration and
the reviewed source inventory. No documentation frontier remains active in this group.

## [Qualification](../glossary.md#term-qualification) and limits

See [the complete qualification record](documentation-redesign-validation.md): all seven
independent learning modes and the 38-line complete transcript, 55 reference classes / 53 main
programs, 78 packages / 43 native components, 542 preserved incoming anchors, supplied/example
[Javadoc](../glossary.md#term-javadoc), actual local reader navigation and 11 accessible diagrams in light/dark.

Independent automated reader/source reviews found and resolved stale semantic statements.
There were no human-reader observations; this is not a usability study. Framework/authority
inputs and [sealed](../glossary.md#term-sealed) private [artifacts](../glossary.md#term-artifact) remain unchanged. The generated local receipt separately
binds final source selection, source-only packaging and exact verification artifacts; this work
does not qualify a new [backend](../glossary.md#backend) or authorize public [distribution](../glossary.md#term-distribution).
