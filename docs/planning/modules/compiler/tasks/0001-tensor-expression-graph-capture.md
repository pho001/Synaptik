# Task 0001: Tensor Expression Graph Capture

## Status

Complete

## Goal

Implement the first bounded compiler capability: capture one non-empty ordered selection of
public `Tensor` expression results into an immutable `CompiledGraphModel`.

Capture translates model-owned pre-capture identities into graph-local identities:

```text
Tensor leaf identity                    -> graph input ValueId
TensorProducer identity                 -> one CompiledNode and NodeId
TensorProducer output position          -> one GraphValue and ValueId
requested Tensor identity and position  -> ordered graph output ValueId
```

The result is structural forward graph capture only. It preserves the current expression DAG,
producer occurrence boundaries, ordered input positions, every declared producer output slot,
and exact descriptor and operation references. It does not infer, prove, transform, optimize,
differentiate, plan, prepare, emit diagnostics, or execute the graph.

## Scope

- Replace the compiler placeholder with one package-private `GraphCapture` implementation in
  `io.github.pho001.synaptik.compiler`.
- Give that type one package-private static entry point with the exact shape:

  ```java
  static CompiledGraphModel capture(List<Tensor> outputs)
  ```

- Add meaningful source Javadoc for the package-private type and entry point. The method contract
  must document its ordered input, immutable result, nullability, identity and ordering semantics,
  and every expected request-validation failure with complete `@param`, `@return`, and `@throws`
  tags.
- Accept a non-null, non-empty ordered list of requested public Tensor results.
- Traverse provenance from the requested outputs through exact `TensorProducer` and input
  `Tensor` references, using object identity rather than structural equality.
- Treat each reachable provenance-free Tensor as one graph input value, deduplicated by exact
  Tensor identity.
- Treat each reachable `TensorProducer` as one computation occurrence, deduplicated by exact
  producer identity even when multiple requested results or consumer paths reach it.
- Emit producers after all reachable input producers, so `CompiledGraphModel.nodes()` is in
  deterministic topological order.
- Assign graph-local IDs independently for every capture call:
  - `ValueId` values begin at zero and increase by one in value-allocation order;
  - `NodeId` values begin at zero and increase by one in node-emission order.
- Define value-allocation order as the deterministic depth-first encounter induced by requested
  output order and each producer's input-position order: a first-seen provenance-free leaf is
  allocated when encountered, and all output positions of a producer are allocated in producer
  output order when that producer is emitted after its inputs.
- Build `CompiledGraphModel.values()` in that same `ValueId` allocation order.
- Build `CompiledNode.inputs()` in exact producer input-position order, preserving repeated input
  positions as repeated `ValueId` references.
- Build every `CompiledNode.outputs()` from all producer output positions in exact descriptor
  order, including slots that have no public result wrapper or are not requested graph outputs.
- Preserve the exact `Operation` reference on the corresponding `CompiledNode` and the exact
  `TensorDescriptor` reference from every producer output slot on its `GraphValue`.
- Map a produced Tensor through its `TensorProvenance.outputIndex()` to the matching producer
  output `ValueId`; do not create a second graph value for the Tensor wrapper.
- Build `CompiledGraphModel.inputs()` in first provenance-free leaf encounter order.
- Build `CompiledGraphModel.outputs()` in caller request order.
- Classify every captured node as `GraphPhase.FORWARD`.
- Preserve zero-input state producers and opaque state edges. In particular, capture through a
  dropout result must retain the reachable `GraphRngKind.INITIAL_STATE` producer, the dropout
  producer's internal state input, all dropout output slots including the non-public mask slot,
  and the requested result's exact output-position mapping.
- Add focused compiler tests for the complete ordering, identity, multi-output, boundary,
  validation, state-edge, immutability, and repeatability contract.
- Update the Compile API reference after implementation to mark only this package-private graph
  capture step current and keep every public compiler lifecycle surface planned.
- Complete the required separate documentation-focused review in the same overall change.

## Out of scope

- any public compiler facade, public capture entry point, transitional API, `GraphCompiler`,
  `CompiledGraph`, engine facade, or compiler service
- a `CompileConfig` aggregate or consumption of current config values
- inference, deferred-constraint representation, binding, equality proof, operand-domain
  revalidation, shape inference, data-type inference, or descriptor rewriting
- graph canonicalization, common-subexpression elimination, interning, dead-code elimination,
  constant folding, algebraic simplification, optional optimization, or any other transformation
- automatic differentiation (autograd), gradient rules, saved-value selection, backward graph
  construction, backward phases, or training-step expansion
- publication binding derivation, `PublicationPlan`, publication policy, or Tensor-to-graph
  publication orchestration
- backend capability analysis, hard eligibility, cost or scoring, owner-map assembly,
  partitioning, logical-memory orchestration, backend ownership, or backend selection
- `CompileArtifacts`, compile diagnostics, public compile failures, or artifact aggregation
- trace payloads, trace event emission, trace correlation allocation, rejection taxonomies, or
  serialization
- prepare, runtime, engine, backend, lowering, kernel, executable, schedule, physical-memory,
  transfer, residency, publication-delivery, or execution behavior
- dependencies, shared Gradle configuration, compiler Gradle configuration other than the narrow
  package-level Javadoc visibility needed to document this task's internal contract, architecture
  rules, ADRs, architecture tests, backend conformance tests, integration tests, or another
  module's source/tests
- planning orchestration or changes to the current package-private planning operations
- sibling-output lookup, generic tuple APIs, output Tensor synthesis, or mutation of model-owned
  Tensor/provenance/producer state
- accepting `GraphRngState` directly as a requested public graph boundary
- creating a detailed Compiler 0002 or later task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the core invariants,
  compiler responsibilities, dependency rules, and compile lifecycle
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Planning contract closure audit](../../planning/planning-contract-closure-audit.md)
- [Model capability closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model task 0009: Compiled graph model](../../model/tasks/0009-compiled-graph-model.md)
- [Model task 0018L: Shared multi-output Tensor provenance](../../model/tasks/0018l-shared-multi-output-tensor-provenance.md)
- [Model task 0019B: Explicit graph RNG state foundation](../../model/tasks/0019b-explicit-graph-rng-state-foundation.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Public API](../../../../api/public-api.md)

## Architecture constraints

- `Tensor` remains public mutable API state and is not graph IR. Capture reads its immutable
  identity-bearing provenance and descriptor facts but stores no Tensor reference in the result.
- `TensorProducer` remains model-owned pre-capture occurrence identity. One exact reachable
  producer maps to one graph node; structurally equal but distinct producers remain distinct.
- `TensorProvenance.outputIndex()` selects one output position of its exact producer. The graph
  node still contains every producer output position, including hidden auxiliary or state slots.
- `CompiledGraphModel`, `CompiledNode`, `GraphValue`, `NodeId`, and `ValueId` remain model-owned
  immutable compile-time data. Compiler owns their allocation and construction during capture.
- Every captured node is forward work. Backward phases require later compiler-owned autograd.
- Capture must be deterministic from requested output order, producer input order, output-slot
  order, and identity relationships. It must never depend on hash-map iteration order, identity
  hash-code values, Tensor IDs, or JVM-global allocation history.
- The compiler produces no physical, prepared, backend-specific, runtime, or executable state.
- No dependency or module-boundary change is authorized. Compiler continues to avoid runtime,
  prepare, engine, and concrete-backend dependencies.
- `ARCHITECTURE.md` remains unchanged. Stop if implementation needs an architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.compiler` — owns the package-private capture implementation and its
  focused same-package tests.
- `io.github.pho001.synaptik.model.tensor` — supplies current public Tensor, producer, provenance,
  descriptor, and opaque RNG-state expression contracts; it is read but not changed.
- `io.github.pho001.synaptik.model.graph` — supplies the immutable graph DTOs constructed by
  capture; it is read but not changed.

Packages added or changed:

- No package is added. The existing compiler root package changes from a placeholder to one
  cohesive internal capture operation.

Type placement:

- `io.github.pho001.synaptik.compiler.GraphCapture` — package-private stateless compiler-owned
  translation from public Tensor expression provenance to `CompiledGraphModel`; it is internal
  because no current cross-package consumer requires an isolated capture API.
- `io.github.pho001.synaptik.compiler.GraphCaptureTest` — same-package focused contract tests that
  can call the package-private entry point without widening production visibility.

`CompilerModule` is removed rather than retained beside real compiler behavior. No replacement
public marker or facade is added.

## Required capture contract

### Request validation and failure order

The entry point validates before traversal in this exact order:

1. reject a null `outputs` reference with `NullPointerException("outputs")`;
2. reject an empty list with `IllegalArgumentException("outputs must not be empty")`;
3. scan all positions in caller order and reject the first null Tensor with a message such as
   `outputs[2]`;
4. scan all positions in caller order by exact Tensor identity and reject the first later repeated
   Tensor reference with a message such as `outputs[2] duplicates outputs[0]`.

After traversal maps requests to logical values, reject the first later requested position that
resolves to a `ValueId` already selected by an earlier position. This second check protects the
unique `CompiledGraphModel.outputs()` contract if future model-owned construction ever exposes two
Tensor wrappers for one producer output slot. Its message must identify both request positions and
the duplicate `ValueId`.

No caller collection is mutated. Capture accepts a mutable list as an input snapshot for the
duration of the synchronous call; only the resulting graph's immutable collections escape.

Supported model construction creates an acyclic expression DAG because producers retain already
existing input Tensors. Behavior for reflectively corrupted cycles or invalid model objects is
not a public capture contract and requires no reflective test fixture.

### Identity and deduplication

- Use reference identity for Tensor leaves and Tensor producers. Value-based `equals`, Tensor ID,
  producer operation equality, descriptors, labels, storage, and identity hash codes do not merge
  occurrences.
- Repeated uses of one exact provenance-free Tensor map to one graph input `ValueId`, while every
  repeated input position remains present in the consuming node.
- Repeated reachability of one exact producer maps to one `NodeId` and one ordered list of output
  `ValueId` values.
- Distinct producers remain distinct nodes even when their operations, inputs, and descriptors
  are structurally equal.
- Multiple public outputs from one producer, such as top-K or attention output-plus-weights, map
  their provenance indices to distinct slots of one node rather than duplicating that node.
- Hidden slots are graph values even when no Tensor wrapper is publicly returned or later
  consumed. Their presence follows `TensorProducer.outputDescriptors()`, not reachability through
  output Tensor wrappers.

### Ordering and identifier allocation

Traversal is deterministic depth-first postorder over producer occurrences:

1. start requested Tensors in caller order;
2. when a produced Tensor first reaches an unseen producer, visit its producer inputs in exact
   input-position order;
3. allocate a provenance-free leaf value when first encountered;
4. after all inputs are available, allocate every producer output value in output-position order,
   then allocate and emit the producer node;
5. skip already-completed producers while reusing their existing output-slot IDs.

The implementation must not consume the Java call stack in proportion to expression depth. Use
an explicit traversal stack or another bounded-stack approach and include a deep-chain regression
test.

The first allocated graph value receives `ValueId(0)` and the first emitted node receives
`NodeId(0)`. Numeric order therefore matches the corresponding stored list order. Each new
capture restarts both graph-local sequences at zero, regardless of Tensor IDs or earlier captures.

No map iteration contributes to `values`, `nodes`, `inputs`, `outputs`, node output positions, or
phase construction. The phase map has structural equality only; its iteration order is not part
of this task.

### Graph boundaries

- A reachable Tensor with absent provenance is a graph input, whether or not it currently has a
  label or host storage. Capture retains only its exact descriptor reference in `GraphValue`.
- A reachable zero-input producer is a source node, not a graph input. This distinction keeps
  `GraphRngState.initial` and other semantic sources inside the graph.
- Graph inputs follow first encounter during the specified traversal. Unreachable Tensors are not
  included.
- Requested graph outputs follow caller order exactly. A pass-through provenance-free Tensor may
  be both a graph input and graph output in a valid zero-node graph.
- Repeating an exact requested Tensor is an error; capture does not silently collapse, reorder, or
  duplicate graph boundary values.
- Direct `GraphRngState` boundary selection remains a current limitation: its state Tensor is
  intentionally package-private to model, and this task adds no state unwrapping API. State is
  captured when it appears on a reachable producer input edge.

### Multi-output and opaque state preservation

For each producer, construct one `GraphValue` for every descriptor position before constructing
the corresponding `CompiledNode`. Node output position `i`, producer descriptor position `i`, and
provenance `outputIndex == i` must resolve to the same `ValueId` and exact descriptor reference.

Tests must exercise at least:

- one single-output chain;
- fan-out from one shared producer and repeated use of one exact leaf;
- two requested public outputs from one genuine shared producer;
- one genuine producer with additional non-public output slots;
- requested outputs presented in an order different from producer output-slot order;
- two structurally equal but identity-distinct producer occurrences;
- one dropout graph that proves preservation of the zero-input initial-state node, the exact
  internal state edge, all three dropout outputs, the hidden BOOL mask slot, and requested result
  slot mapping.

The capture result retains no public Tensor, `TensorId`, `TensorProducer`, `TensorProvenance`, or
`GraphRngState` reference. Identity maps are construction-local and do not escape.

## Affected files

Expected implementation paths:

- remove
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompilerModule.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCaptureTest.java`
- update `modules/compiler/build.gradle.kts` only so the compiler Javadoc task includes
  package-private declarations without changing Java visibility
- update [Compile API](../../../../api/compile-api.md)
- update and finalize this task specification
- update [Compiler master plan](../master-plan.md)
- update [Roadmap](../../../roadmap.md)

Review without modification unless a documented conflict requires stopping: model source/tests,
Tensor API, Public API, glossary, architecture documents/tests, root/shared Gradle files, config,
trace, planning, runtime, prepare, engine, backend, conformance, and integration paths.

## Maximum scope

This task may create, modify, or remove at most the exact eight paths listed under
[Affected files](#affected-files).

No other production type, test suite, documentation page, planning file, build file, module, or
architecture path is authorized. If accurate implementation or documentation requires a ninth
path, stop and propose a separately justified follow-up instead of silently expanding scope.

## Acceptance criteria

- `CompilerModule` is removed and no placeholder or transitional public compiler type replaces
  it.
- The compiler production package contains one package-private final `GraphCapture` type with one
  package-private static `capture(List<Tensor>)` entry point returning only
  `CompiledGraphModel`.
- `GraphCapture` and its entry point have meaningful source Javadoc that explains the capture
  boundary rather than restating the declarations; the method documents its input constraints,
  immutable result semantics, and expected failures with complete `@param`, `@return`, and
  `@throws` tags.
- Capture validation and failure ordering match the required contract exactly.
- Graph-local `ValueId` and `NodeId` sequences restart at zero per call and follow the specified
  deterministic allocation order.
- Graph values retain exact source descriptor references, nodes retain exact producer operation
  references, and no model identity object is retained in the finished graph.
- Provenance-free Tensor leaves are identity-deduplicated graph inputs in first encounter order.
- Producer identities are deduplicated, structurally equal distinct producers remain distinct,
  and repeated input positions remain repeated node inputs.
- Nodes are topologically ordered without recursion proportional to expression depth.
- Every producer output descriptor creates one ordered graph value and node output, including
  unused, auxiliary, hidden, and opaque state positions.
- Requested Tensor results map through provenance output indices to graph outputs in caller order;
  duplicate requested identities and duplicate resolved logical values fail as specified.
- Every node is classified `GraphPhase.FORWARD`.
- Focused tests cover single-output, pass-through, shared subgraphs, repeated positions,
  multi-output ordering, hidden outputs, opaque RNG-state edges, distinct equal occurrences,
  deterministic repeat capture, deep chains, immutability, and every request failure.
- The Compile API describes package-private capture as current without claiming a public compile
  API, inference, validation beyond structural graph construction, transformation, autograd,
  planning, diagnostics, prepare, backend support, runtime behavior, or execution.
- Tensor API, Public API, glossary, architecture docs/tests, other modules, shared Gradle
  configuration, conformance, and integration tests remain unchanged for the recorded reasons.
- Compiler Javadoc generation includes package-private declarations without changing source
  visibility, Java behavior, dependencies, or Javadoc policy for another module.
- Exactly the eight authorized paths change, no later compiler task specification exists, and only
  Compiler 0001 is `Ready` before implementation or `Complete` after all evidence is final.
- A separate documentation-focused agent pass has finalized affected documentation, Javadocs,
  terminology, links, status, and glossary impact in this same overall change.

## Tests / validation

During implementation, run the focused suite as needed:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCaptureTest
```

After executable code stabilizes, run one final compiler module suite:

```bash
./gradlew :modules:compiler:test
```

The implementation context records the exact outcomes and hands them to the documentation-focused
agent. The documentation pass must not repeat successful Java tests unless it changes executable
Java behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must also check local Markdown links and anchors, balanced fences, final
newlines, terminology, exact eight-path scope, removal of `CompilerModule`, package-private
capture visibility, source-Javadoc completeness for the new type and entry point, exactly one
current detailed compiler task, synchronized task/master/roadmap status, and absence of a Compiler
0002 or later specification.

Repository-wide validation is deferred to the compiler capture-and-validation capability
checkpoint or continuous integration. This task changes one module's internal implementation and
module-local Javadoc visibility, with no dependency, shared build configuration, architecture
boundary, or public API change.

## Dependencies

- Model task 0009, immutable compiled graph model — Complete.
- Model task 0018L, identity-based shared multi-output producer provenance — Complete.
- Model task 0019B and its completed dropout follow-up, explicit opaque graph RNG state and state
  edges — Complete.
- Model task 0024A, selected model capability milestone closure — Complete.
- Planning task 0006 and its `CLOSED` contract audit — Complete; capture does not yet invoke its
  package-private evaluators/generators.
- Current compiler module dependency direction — already architecture-compliant and unchanged.

Config 0004, Trace 0003+, Runtime, and Prepare are not dependencies of this bounded capture task.
They remain Draft because capture consumes none of their future cost, payload, prepared, or
runtime contracts.

## Follow-up tasks

Future Draft compiler rows, in master-plan order:

- 0002 — captured-graph operand revalidation, inference, and deferred-constraint proof.
- 0003 — canonicalization and forward optimization after validation semantics are stable.
- 0004 — autograd and backward graph construction after forward graph contracts are validated.
- 0005 — publication and planning orchestration plus immutable `CompileArtifacts` after compiler,
  planning, config, trace, and downstream consumer contracts are ready.

These are concise Draft master-plan rows only. Do not create their detailed specifications in
this task or implement any part of them opportunistically.

## Architecture impact

Expected impact: None.

This task implements the existing architecture-defined `Tensor output -> GraphCapture ->
topological graph` lifecycle inside `modules/compiler`. It introduces no public surface, module
dependency, ownership change, prepared state, or executable behavior. If implementation requires
an architecture, dependency, or public-facade decision, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler master plan, and
docs/planning/modules/compiler/tasks/0001-tensor-expression-graph-capture.md. Read the directly
referenced model source/tests and current Compile/Tensor/Public API boundaries needed to verify
the task.

Implement Compiler 0001 exactly as specified within its eight authorized paths. The only build
change permitted is compiler-local Javadoc inclusion of package-private declarations. Do not
implement out-of-scope inference, proof, optimization, autograd, publication/planning orchestration,
CompileArtifacts, diagnostics/tracing, public facades, prepare/runtime/backend/engine work, or
dependency/shared-build changes. Stop on any scope or architecture conflict.

After code implementation and the final compiler test evidence, hand the actual diff and evidence
to a separate documentation-focused agent or thread with clean context. That targeted pass must
follow docs/developer-guide/documentation-rules.md, finalize Javadocs, Compile API, planning
status/evidence, terminology, links, and glossary impact in the same overall change, and must not
repeat successful Java tests unless executable behavior changes or a concrete risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes, completion
summary, and final status. Do not mark it Complete before the documentation pass and every
acceptance criterion finish.
```

## Local decisions

- Use one package-private static capture boundary because the first consumer is the focused
  same-package test. A public compiler or planning collaboration is not yet justified.
- Allocate IDs from deterministic traversal order rather than Tensor IDs, identity hash codes, or
  model allocation history. IDs are graph-local and restart for each capture.
- Producer output slots, not only public result wrappers, define node outputs. This preserves
  current genuine multi-output, hidden saved-value, and opaque state positions without adding a
  sibling-output API.
- Reject duplicate requested boundaries rather than silently collapse them, because
  `CompiledGraphModel.outputs()` is ordered and unique and silent normalization would erase caller
  intent.
- Keep direct `GraphRngState` boundary selection unsupported. Its private Tensor boundary is an
  intentional model API constraint; reachable state edges remain fully capturable through
  producer inputs.
- Use an explicit traversal stack so valid deep expression graphs do not depend on Java call-stack
  depth.

## Known limitations

- Only forward capture is implemented; every node phase is `FORWARD`.
- The result is package-private implementation output and has no public compile facade or config
  aggregate.
- Capture trusts current model construction invariants for descriptor/provenance agreement and an
  acyclic producer graph. It adds no reflective-corruption contract.
- Produced descriptors are preserved rather than inferred or revalidated. Dynamic or expression
  Shape obligations remain unresolved.
- All producer output slots remain in the graph even when unused. This task performs no dead-code
  elimination or saved-value liveness decision.
- `GraphRngState` cannot be selected directly as a graph boundary because it intentionally exposes
  no public Tensor; state is captured only when reachable on a producer edge.
- No public mapping from Tensor/TensorProducer identities to graph-local IDs escapes capture.

## Validation evidence

- Implementation context ran
  `./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCaptureTest`
  after the initial executable implementation. It passed with `BUILD SUCCESSFUL in 1s`; 13 Gradle
  tasks were considered, 3 executed and 10 were up-to-date.
- After removing one unused test import, the implementation context reran
  `./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCaptureTest`.
  It passed with `BUILD SUCCESSFUL in 905ms`; 13 Gradle tasks were considered, 2 executed and 11
  were up-to-date.
- After executable Java stabilized, the implementation context ran exactly one final
  `./gradlew :modules:compiler:test`. It passed with `BUILD SUCCESSFUL in 980ms`; 13 Gradle tasks
  were considered, 1 executed and 12 were up-to-date. The final XML report records 12 tests, 0
  skipped, 0 failures, and 0 errors for `GraphCaptureTest`.
- Repository-wide validation remains deferred to the named compiler capability checkpoint or CI
  because this task changes one module's internal implementation and no dependency, shared build
  configuration, architecture boundary, or public API.
- The initial separate clean-context documentation pass was performed by Codex task identity
  `/root` on 2026-07-15, acting only as the mandatory documentation-focused review agent. It
  applied the
  General, API/Javadoc, and Planning documentation profiles; it also reviewed the Example profile,
  but added no new runnable or conceptual example requiring example validation.
- That pass independently reviewed `AGENTS.md`, `ARCHITECTURE.md`, the current architecture index,
  documentation rules and selected profiles, the planning guide, roadmap, compiler master plan,
  this task, the actual combined diff, removed `CompilerModule`, final `GraphCapture` source and
  test, the final XML test report, directly relevant Tensor/producer/provenance/RNG/graph source,
  generated/current Compile API, relevant Tensor API, Public API, and glossary boundaries.
- No executable Java behavior or test changed during documentation review. The pass therefore
  reused the implementation context's final `./gradlew :modules:compiler:test` evidence: 12 tests,
  0 skipped, 0 failures, and 0 errors. Inspection of
  `modules/compiler/build/test-results/test/TEST-io.github.pho001.synaptik.compiler.GraphCaptureTest.xml`
  confirmed those exact counts and all 12 named cases in the final report.
- The pass finalized the package-private `GraphCapture` type and method Javadocs and updated the
  Compile API so only identity-based package-private structural forward capture is current. Every
  public compiler facade and lifecycle, inference/revalidation/proof, transformation/optimization,
  autograd/backward construction, publication/planning orchestration, diagnostics,
  `CompileArtifacts`, tracing/emission, prepare/backend/runtime, physical-memory, and execution
  behavior remains explicitly planned.
- `./gradlew :modules:compiler:javadoc` failed with exit code 1 and
  `error: No public or protected classes found to document.` The failure occurs because removing
  the public placeholder leaves only the intentionally package-private `GraphCapture`, while the
  standard Gradle Javadoc task does not include private/package-private declarations. Resolving it
  requires either widening the Java API, which this task forbids, or changing Javadoc/build
  configuration outside the exact seven authorized paths. Neither change was made.
- As a diagnostic rendering only, `javadoc -private
  @modules/compiler/build/tmp/javadoc/javadoc.options` passed with exit code 0 and three expected
  missing-comment warnings for private implementation details (`TraversalFrame`, the private
  constructor, and `resolve`). Inspection of generated
  `modules/compiler/build/docs/javadoc/io/github/pho001/synaptik/compiler/GraphCapture.html`
  confirmed the finalized type purpose, package-private boundary, `FORWARD` phase statement,
  parameter constraints, immutable return semantics, and both documented exception categories.
- `python3 /tmp/validate_synaptik_markdown.py` passed: 232 Markdown files, 4,159 local links,
  252 local anchors, 2,930 fence markers, and final-newline/trailing-whitespace validation all
  passed.
- `git diff --check` passed. The combined
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` output contained
  exactly the seven authorized paths, and `git status --short` reported only those same paths.
- The exact shell audit for seven-path count, deleted placeholder, one detailed task, no Compiler
  0002-or-later specification, task/master/roadmap status, package-private type and method, absence
  of a public capture declaration, and absence of placeholder source passed with
  `scope/status/no-later-spec/no-public-capture/no-placeholder checks=pass`.
- The first source-Javadoc/current-planned boundary audit command failed because its grep patterns
  expected one leading space before Javadoc tags instead of the source's five spaces. This was a
  validation-command pattern error, not a source failure. The corrected exact audit found one
  `@param outputs`, one `@return`, both required `@throws` categories, the single current internal
  capture section, and the explicit no-public/no-runnable/conceptual-planned boundaries; it passed
  with `source-Javadoc/current-planned-boundary checks=pass`.
- Tensor API required no change because its public Tensor, producer, provenance, and graph DTO
  contracts remain accurate; model construction still performs no capture itself, while the new
  compiler operation is package-private and changes no model API. Public API required no change
  because no public compiler declaration or callable lifecycle was added. The glossary required
  no change because no reusable term changed meaning: `CompiledGraphModel` still does not perform
  capture itself, and producer/provenance remain pre-capture identities.
- Architecture documents and architecture tests required no change because task 0001 implements
  the already-authorized compiler capture responsibility without changing dependencies or module
  boundaries. Backend conformance and integration tests required no change because the task adds
  no backend or end-to-end execution behavior. Other modules and their source/tests required no
  change because capture consumes their existing contracts without altering them.
- Gradle configuration required no semantic or dependency change for the implementation. However,
  the exact compiler Javadoc failure above is a concrete build-documentation validation gap that
  cannot be corrected within this task's authorized paths. No Gradle file was changed.
- The narrowly authorized follow-up implementation context `/root/fix_compiler_javadoc` expanded
  the task scope by one path, `modules/compiler/build.gradle.kts`, and configured only its standard
  `javadoc` task with `JavadocMemberLevel.PACKAGE`. Production and test Java source, dependencies,
  root/shared Gradle conventions, other modules, and source visibility remained unchanged.
- That context ran `./gradlew :modules:compiler:javadoc` after the configuration stabilized. It
  passed with `BUILD SUCCESSFUL in 1s`; 7 actionable tasks were considered, 1 executed and 6 were
  up-to-date. The generated options contain `-package`, the generated class index and
  `GraphCapture.html` contain the package-private type and `capture(List)` method, and the page
  excludes the private constructor, `resolve` helper, and `TraversalFrame` implementation detail.
- The same context then ran exactly one `./gradlew :modules:compiler:test` after the build
  configuration stabilized. It passed with `BUILD SUCCESSFUL in 957ms`; 13 actionable tasks were
  considered, 1 executed and 12 were up-to-date. The final XML report still records 12 tests, 0
  skipped, 0 failures, and 0 errors for `GraphCaptureTest`.
- The follow-up's final `git diff --check` passed. `git diff --name-status` contained only
  the task specification and compiler build file, while `git diff HEAD^ --name-status` contained
  exactly the task's eight authorized paths across the original capture commit and this follow-up.
  The tasks directory still contains only `.gitkeep` and the 0001 specification; no 0002-or-later
  detailed task exists.
- The final separate clean-context documentation pass was performed by Codex task identity
  `/root/finalize_compiler_0001` on 2026-07-15. It independently applied the General,
  API/Javadoc, and Planning profiles and reviewed the Example profile; no example changed. It
  reviewed the authoritative architecture and documentation workflow, planning guide, roadmap,
  compiler master plan and task, Compile API, final source/tests, compiler Gradle configuration,
  generated Javadoc, prior commit/diff, and recorded executable-test evidence.
- No executable Java behavior changed after `/root/fix_compiler_javadoc` ran the final compiler
  test suite, so this pass did not repeat it. The reused final XML report records 12 tests, 0
  skipped, 0 failures, and 0 errors.
- `./gradlew :modules:compiler:javadoc` passed after final documentation edits with
  `BUILD SUCCESSFUL in 362ms`; 7 actionable tasks were up-to-date. The generated options contain
  `-package`, generated Javadoc contains package-private `GraphCapture` and `capture(List)`, and it
  excludes the private constructor, `resolve` helper, and `TraversalFrame` implementation detail.
- One intervening repeat of that command failed before executing Gradle tasks because the sandbox
  denied access to Gradle's home-directory wrapper lock file. The exact command was rerun with the
  required filesystem approval and produced the successful 362ms result above; this was an
  environmental permission failure, not a source or Javadoc failure.
- `python3 /tmp/validate_synaptik_markdown.py` passed after final planning synchronization: 232
  Markdown files, 4,159 local links, 252 local anchors, 2,930 fence markers, and all final-newline
  and trailing-whitespace checks passed.
- Final scope/status checks passed: `git diff HEAD^ --name-status` contains exactly the eight
  authorized paths; the compiler tasks directory contains only `.gitkeep` and task 0001; task
  0001, its master-plan row, and the roadmap all record `Complete`; tasks 0002–0005 remain Draft
  master-plan rows without detailed specifications; `CompilerModule` is absent; and capture
  remains package-private.
- Final `git diff --check` passed. Final `git status --short` contains only the task specification,
  compiler master plan, roadmap, and compiler build file because the other four authorized paths
  are already present in the immediately preceding implementation commit.
- Tensor API and Public API still require no change because capture remains internal and does not
  alter any model or public declaration. The glossary still requires no change because capture,
  producer/provenance, and `CompiledGraphModel` retain their existing meanings. Architecture
  documentation/tests and Gradle dependencies require no change because ownership, dependency
  direction, and module boundaries are unchanged. Other modules, backend conformance, and
  integration tests require no change because this task adds no backend or end-to-end behavior.

## Implementation notes

- Removed the public `CompilerModule` placeholder and added one package-private final
  `GraphCapture` with the specified package-private static entry point.
- Capture uses construction-local `IdentityHashMap` instances for provenance-free Tensor leaves
  and producer occurrences. An explicit frame stack performs deterministic depth-first postorder
  traversal without Java recursion proportional to expression depth.
- Graph-local IDs are allocated from zero per call. Leaves are allocated on first encounter;
  every producer descriptor slot is allocated before its node; node inputs and requested outputs
  retain their required order and repeated input positions.
- The focused suite covers API visibility, a single-output chain, zero-node pass-through,
  fan-out/repeated positions, shared genuine multi-output ordering, hidden dropout mask and opaque
  RNG-state edges, distinct equal producers, repeat capture and ID restart, collection
  immutability, a 20,000-node chain, exact request-validation order, and the duplicate-resolved
  logical-value guard.
- The implementation retains only exact Operation and TensorDescriptor references in the final
  graph DTOs. Tensor, TensorId, TensorProducer, TensorProvenance, GraphRngState, and identity maps
  remain traversal-local and do not escape.
- No inference, constraint proof, transformation, optimization, autograd, publication/planning
  orchestration, compile aggregate, tracing, public facade, prepare/runtime/backend/engine work,
  dependency, shared-build, architecture, or later-task implementation was added.
- The authorized follow-up changed only compiler-local Javadoc member inclusion from Gradle's
  default public/protected level to package level; it did not change Java behavior, source
  visibility, dependency direction, or another module's documentation policy.

## Completion summary

- Completed changes: implemented the complete package-private graph-capture contract and focused
  executable tests; removed the compiler placeholder; configured compiler-local Javadoc to render
  package-private contracts; and synchronized task, master-plan, and roadmap completion state.
- Files changed or created in the complete task: updated Compile API, compiler master plan, this
  task, roadmap, and compiler build configuration; removed `CompilerModule.java`; and added
  `GraphCapture.java` and `GraphCaptureTest.java`—exactly the eight authorized paths.
- Tests and validation: both focused runs passed; the implementation's stabilized module run and
  the Javadoc-configuration follow-up's final module run each passed, with the final report
  recording 12 tests, 0 skipped, 0 failures, and 0 errors.
- Documentation-agent review: the mandatory final clean-context pass finalized the authorized
  Javadocs, Compile API boundary, terminology, links, evidence, and planning status without
  changing executable behavior.
- Documentation impact: `docs/api/compile-api.md` now distinguishes current internal structural
  forward capture from every planned public or later compiler lifecycle responsibility. Tensor
  API, Public API, glossary, architecture documentation, and other module documentation were
  reviewed and remain unchanged for the recorded reasons.
- Javadoc review: source and generated Javadocs passed inspection; the exact compiler Gradle
  Javadoc task now passes with package-level inclusion while private implementation details remain
  excluded.
- Glossary impact: no new reusable term or changed term boundary; no glossary edit required.
- Unresolved issues: None.
- Follow-up required: None. Compiler tasks 0002–0005 remain Draft and require their own future
  planning steps.

Status: Complete
