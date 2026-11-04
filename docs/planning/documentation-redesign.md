# Didactic documentation redesign

## Status and authorization

Status: Draft — user-requested follow-on work, queued after the active public-readiness audit.
The user approved a reader-oriented structure and explicitly required an explanation of the whole
framework, every package and commented examples with explanations. No implementation is launched
by this record. Existing normative authority and numerical/ownership contracts do not change.

Depends on: the source-matched public-readiness audit and its current capability/distribution
receipt. Conflicts with: active audit documentation edits. Parallel group: None until a Ready
execution split assigns exclusive files. Common base revision: select the completed audit join
at activation. Integration order: reader entry/one complete learning journey, then topical guides,
framework explanation/package atlas, reference and old-link migration. Shared-document integration
owner: coordinator. Integration validation: runnable external public-API examples, documentation
link/anchor validation, affected Javadoc and human learning-journey review. Classification: A for
editorial changes; B/C if a later executable/API/ownership boundary is actually changed.

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

## Activation

After audit closure, reconcile the approved proposal with current sources and documentation,
choose the audited base, inventory packages/examples/links, and create bounded Ready frontiers
with explicit file ownership. Only then launch editorial/executable-example work.
