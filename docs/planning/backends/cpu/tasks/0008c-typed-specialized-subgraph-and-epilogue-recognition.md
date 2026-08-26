# Task 0008C: Typed Specialized-Subgraph and Epilogue Recognition

## Status

Complete

## Goal

Add one closed, CPU-private, typed cold-analysis contract that recognizes selected MATMUL,
channels-first convolution, and floating-reduction epilogue subgraphs plus the exact first-class
softmax and normalization kernels that the CPU already executes. Preserve the complete ordinary
CPU 0008B decomposition as the deterministic execution plan whenever recognition is absent,
ineligible, ambiguous, unsupported, or not backed by an already implemented specialized form.

This task adds recognition facts only. It does not add a new numerical algorithm, generated body,
artifact schema, capability claim, or executable family. In particular, MATMUL remains
fail-closed until CPU 0008F, and recognized Conv1d, Conv3d, or reduction epilogues retain their
already implemented materialized split units. The only recognized multi-node form that may point
at an existing fused generated artifact is the exact Conv2d `ADD` or `ADD -> RELU` form already
implemented and evidenced by CPU 0008.

## Mental model

Recognition describes a bounded opportunity; it does not execute it:

```text
complete projected CPU partition
  -> deterministic typed recognition facts
  -> independently prove the CPU 0008B split baseline
  -> associate each fact with its exact baseline units
  -> select only an already implemented identical specialized form
  -> otherwise keep the exact split baseline and artifacts
```

The distinction is deliberate:

```text
recognition identity   = stable CPU-private semantic/topology facts for later comparison
artifact identity      = existing generated-code structural identity, unchanged by this task
execution              = existing prepared units only
```

CPU 0008D may later compare complete legal fused and split candidates. CPU 0008F may later add
MATMUL execution. Neither permission is implied by recognizing a shape today.

## Scope

### Closed typed recognition model

- Add one immutable sealed CPU-private recognition fact family with exactly four variants:
  `MatmulEpilogue`, `ConvolutionEpilogue`, `ReductionEpilogue`, and `ExplicitSemanticKernel`.
  Use records and enums, not strings, reflective annotations, maps, registries, callbacks, or a
  domain-specific language (DSL).
- Every fact records:
  - one closed family/form enum;
  - exact ordered member-node ordinals and exact associated baseline-unit indices;
  - exact ordered input/result data types;
  - exact static Shapes and resolved access-regime facts needed by its eligibility checks;
  - exact anchor attributes or a typed immutable structural projection of them;
  - intrinsic-bias presence where applicable;
  - one closed epilogue topology;
  - one execution disposition; and
  - one stable typed structural identity that excludes graph, node, value, partition, slot,
    carrier-object, address, run, class-loader, and artifact-store identity.
- The exact execution dispositions are:
  - `EXISTING_SPECIALIZED` — the recognized form maps to byte-identical already implemented
    specialized execution;
  - `ORDINARY_SPLIT` — recognition is retained, but the exact CPU 0008B baseline units and
    materialized boundaries execute; and
  - `UNSUPPORTED_ANCHOR` — a typed fact may be returned by the focused recognizer for inspection,
    but ordinary CPU preparation remains fail-closed because the anchor has no implemented CPU
    execution family.
- `UNSUPPORTED_ANCHOR` is permitted only for MATMUL recognition. It must never make
  `CpuCapabilityProvider.supports(...)` true, create a seed, produce a resource declaration,
  reach finalization, or appear in a successfully prepared plan before CPU 0008F implements the
  exact anchor.
- The structural identity is a recognition/candidate identity for later CPU-private reasoning,
  not a generated-artifact key. Existing `CpuPortableKernelIr.structuralKey()`,
  `CpuLoweringFingerprint`, `CpuKernelSpecialization`, generator schema 52, and generated class
  bytes remain authoritative and byte-identical for every selected executable unit.

### Uniform bounded epilogue vocabulary

- The only admitted external epilogue topology is:

  ```text
  anchor
    -> optional BinaryArithmeticKind.ADD with one external operand
    -> optional terminal exact unary activation or ScalarElementwiseKind.CLAMP
  ```

- The closed terminal activation set is `RELU`, `SIGMOID`, `TANH`, `GELU`,
  `GELU_TANH_APPROXIMATION`, and `SILU`, plus first-class scalar `CLAMP` with its exact typed
  `ClampRangeAttrs`. These are recognition labels only; they do not broaden a family emitter.
- The optional ADD must consume the immediately preceding anchor result in exactly one operand
  position. Its other operand must be external to the recognized members, and its exact current
  descriptor must right-broadcast to the anchor result Shape. ADD input order is retained because
  it is part of the graph even though arithmetic addition is commutative.
- Every admitted epilogue operation must produce the same Shape and data type as its immediately
  preceding result, use an already CPU-supported exact pointwise meaning, and have a resolved
  injective output layout. No cast, scalar add/multiply, binary multiply, comparison, WHERE,
  affine view, second ADD, second activation, or other operation is recognized as an epilogue.
- Preserve every represented operation boundary. Recognition never folds ADD into an anchor
  accumulator, combines intrinsic and external bias, reassociates a reduction or contraction,
  erases BFLOAT16/FLOAT32 narrowing, changes fused-multiply-add permission, or moves activation
  across another operation.

### MATMUL epilogue recognition

- Recognize exactly one `MatmulKind.MATMUL` occurrence with `NoOperationAttrs.INSTANCE`, two
  ordered inputs, and one output whose current Compiler-validated descriptors obey Model 0019's
  vector/matrix/batched rank, contraction, right-aligned batch broadcast, promotion, result Shape,
  and metadata rules.
- Recognize all current Model MATMUL data-type categories in the typed fact: BFLOAT16, FLOAT32,
  FLOAT64, INT32, and INT64 where the exact occurrence is valid. BOOL and any future type are
  ineligible. The optional ADD/activation/CLAMP suffix is admitted only for FLOAT32 or FLOAT64
  result types because those are the current exact common pointwise/activation execution types;
  other MATMUL facts have epilogue `NONE`.
- Record whether the right MATMUL input is produced by an exact single-use, unpublished rank-two
  `PERMUTE` with axes `[1, 0]`. That predecessor is a typed `TRANSPOSED_WEIGHT` input form, but it
  remains outside the epilogue member list and remains an ordinary affine-view seed. No other
  permutation, rank edit, or weight orientation is inferred as linear.
- An optional ADD is `LINEAR_BIAS` only when its external operand is exact rank one and its sole
  Dimension equals the MATMUL result's final Dimension. Other exact right-broadcast ADD operands
  are not recognized for MATMUL in this task.
- Every MATMUL fact has execution disposition `UNSUPPORTED_ANCHOR`. Direct focused recognition
  tests may inspect it, but complete CPU preparation must fail at the same independently
  unsupported seed boundary as CPU 0008B. CPU 0008F owns capability, lowering, generated
  execution, safe split/fusion, schema changes, and performance evidence.

### Convolution epilogue recognition

- Recognize exactly these anchors:
  - the CPU 0008A four-node visible NCW Conv1d composition, including both axis-2 expansions,
    grouped Conv2d attributes, axis-2 squeeze, virtual singleton views, and optional intrinsic
    rank-one bias;
  - one first-class grouped NCHW `CONV2D` occurrence; and
  - one first-class grouped NCDHW `CONV3D` occurrence.
- Reuse the completed rank-specific validation. Do not introduce `ConvNd`, dynamic rank, another
  convolution geometry owner, or looser similar-looking Conv1d recognition.
- Apply the uniform optional ADD plus optional terminal epilogue only to FLOAT32/FLOAT64 anchor
  results. BFLOAT16 remains direct/intrinsic-bias-only because current CPU pointwise arithmetic
  does not execute BFLOAT16. Intrinsic bias is anchor state and is never the optional external
  ADD.
- An exact Conv2d `ADD` or `ADD -> RELU` already accepted by `CpuConv2dLowering` is
  `EXISTING_SPECIALIZED` and must retain its existing `CpuConv2dIr.Epilogue`, structural key,
  class bytes, algorithm, operation boundaries, and evidence.
- Conv2d with another recognized terminal activation/CLAMP, Conv1d with any epilogue, and Conv3d
  with any epilogue are `ORDINARY_SPLIT`. They retain the exact established anchor seed followed
  by the exact ordinary pointwise seed(s), ordinary split buffers, unit dependencies, resources,
  and sequential composite execution from CPU 0008B.
- Recognition never changes requested Conv1d intermediate visibility, Conv2d/Conv3d intrinsic
  bias meaning, convolution traversal, accumulation format, padding multiplication, grouping,
  narrowing, output-cell ranges, workspaces, aliases, or publications.

### Floating-reduction epilogue recognition

- Recognize exactly one current first-class floating reduction anchor from this closed set:
  `SUM`, `MEAN`, `PROD`, `MIN`, `MAX`, `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`,
  `L1_NORM`, or `L2_NORM`.
- Admit the exact current full, single-axis, and multi-axis attributes supported by the selected
  kind, including statistical correction where applicable. Exclude `SumToShapeAttrs`,
  `MaskedReductionAttrs`, `ARG_MIN`, `ARG_MAX`, `ALL`, and `ANY` from 0008C recognition. Those
  operations retain their current independent execution and ordinary barriers.
- Require a FLOAT32 or FLOAT64 anchor result and the exact uniform optional ADD plus optional
  terminal epilogue. BFLOAT16 reductions remain independently executable but are not recognized
  with pointwise epilogues until BFLOAT16 pointwise arithmetic exists.
- Every reduction epilogue has execution disposition `ORDINARY_SPLIT`. The reduction remains one
  indivisible numerical-order seed with its established algorithm, accumulator, traversal,
  scratch, empty-domain, exceptional-value, and range contract. Each suffix operation remains an
  ordinary pointwise unit separated by the exact CPU 0008B logical-value buffer.
- Do not recognize a chain of primitive reductions/arithmetic as softmax, log-softmax, Layer
  normalization, RMS normalization, batch normalization, attention, or a loss. A reduction
  epilogue fact names only its explicit reduction anchor and literal suffix operations.

### Explicit semantic-kernel recognition

- Recognize only these already implemented first-class one-node kernels:
  `SOFTMAX`, `LOG_SOFTMAX`, `LAYER_NORM`, `RMS_NORM`, `BATCH_NORM_INFERENCE`, and
  `BATCH_NORM_TRAINING`.
- Reuse their exact current CPU admission and lowering facts, including attributes, ordered
  inputs/outputs, output slots, data types, Shapes, layouts, workspaces, range forms, and
  multi-output roles. Each receives disposition `EXISTING_SPECIALIZED` and exactly one associated
  baseline unit.
- First-class means the actual Model operation kind. Never infer one of these facts from EXP,
  LOG, reductions, arithmetic, square root, reciprocal, affine operations, saved-statistic-like
  values, or any other decomposed graph. Similar mathematics, matching Shapes, constant values,
  and familiar operation order are irrelevant.
- Explicit semantic-kernel recognition has no epilogue in this task. Any consumer remains an
  ordinary CPU 0008B unit and the kernel output remains materialized.

### Deterministic precedence, overlap, and ambiguity

- Recognition is a single bounded forward scan over the supplied stable topological node order.
  It does not rewrite, reorder, clone, or remove a `CompiledNode` or `GraphValue`.
- At each unclaimed anchor ordinal, test families in this exact order:
  `CONV1D_COMPOSITION`, `CONV2D`, `CONV3D`, `MATMUL`, `REDUCTION`,
  `EXPLICIT_SEMANTIC_KERNEL`.
- For one anchor, prefer the longest eligible epilogue: `ADD -> terminal`, then `ADD`, then
  `terminal`, then no epilogue. An ineligible longer form does not consume nodes; the recognizer
  may select the next shorter form only when every member and boundary of that shorter form is
  independently eligible.
- A node may belong to at most one retained recognition fact. A later candidate that overlaps an
  already retained candidate is rejected and remains ordinary. Equal-length candidates that
  would claim the same unclaimed nodes are an ambiguity: retain neither and use ordinary
  decomposition. Do not use hash order, enum text, `ValueId` magnitude, or object identity as a
  tie-breaker.
- The exact associated baseline-unit indices are resolved only after the maximally split CPU
  0008B baseline succeeds. A fact may span multiple consecutive dependency-linked units, but it
  may not cause a unit contraction. Unit-member ordinals must cover every fact member exactly,
  in order, with no unrelated unit member inserted.

### Barriers and exact fallback

- Every internal edge proposed for a multi-node fact must have exactly one in-partition consumer,
  must not be a graph output/publication, and must have complete projected producer/consumer and
  logical-memory facts. Fan-out or publication makes the longer candidate ineligible; the exact
  CPU 0008B materialized topology remains.
- State, random, dropout, ordering, scatter, fold, indexing, scan, affine-copy, multi-output,
  alias-uncertain, unresolved-layout, dynamic-Shape, non-injective-write, and unsupported-type
  nodes are barriers unless they are the exact one-node explicit semantic kernel named above.
- Any unresolved or potentially overlapping recognized-result write versus recognized input,
  external epilogue operand, or other recognized write is a barrier. Recognition may reuse the
  current exact resolved span/access proof; it must not introduce copying or assume future slot
  separation. Existing unit-level cold alias checks remain authoritative even after recognition.
- Candidate failure catches no VM error and hides no malformed graph. Structural ineligibility
  returns no fact; malformed projection, impossible baseline association, or disagreement with a
  selected existing specialized form fails preparation before resource declaration or artifact
  access.
- Unrecognized or ineligible supported graphs must produce a plan exactly equal to CPU 0008B's
  ordinary plan in unit membership, dependency order, buffers, workspaces, carrier/access facts,
  route, specializations, artifact keys, and executable order. Recognition facts are the only
  permitted plan difference.
- A recognized `ORDINARY_SPLIT` graph must also retain that exact plan. No recognized fact is a
  hidden fused candidate until a later task adds an executable form and all corresponding
  legality, resource, schema, Class-File, oracle, and performance evidence.

### Hard recognition budgets

These are fail-closed cold-analysis ceilings, not profitability scores:

| Budget | Ceiling | At ceiling | Over ceiling |
|---|---:|---|---|
| Partition nodes | 8 | Analyze | Existing CPU 0008B partition rejection |
| Anchor ordinals examined | 8 | Analyze | Impossible inside an admitted partition |
| Recognition attempts | 24 | Stop after the twenty-fourth tested anchor/topology form | Retain facts already proved; ordinary fallback for the rest |
| Retained recognition facts | 8 | Eight distinct one-node facts fit exactly | A ninth fact is impossible inside the unchanged eight-node partition ceiling |
| Members in one fact | 6 | Conv1d plus the two-step epilogue fits exactly | Impossible in the closed family grammar |
| Epilogue operations | 2 | Optional ADD plus one terminal | Impossible in the closed suffix grammar |
| Referenced materialized boundary positions recorded by one fact | 10 | A first-class batch-normalization-training node with five distinct inputs and five distinct outputs records ten access facts | An eleventh position is impossible in the closed family matrix |
| Associated baseline units | 2 | One indivisible anchor unit plus one vertically fused pointwise-suffix unit | Impossible under the unchanged CPU 0008B seed and vertical-fusion invariants |

- A referenced materialized boundary position is exactly one entry in the fact's ordered
  `accessFacts`: `CpuSpecializedSubgraphRecognizer.build(...)` appends one access fact for each
  ordered anchor input position, followed by one for each ordered final-result position passed to
  the fact. This is a positional count, not a count of publicly exposed Tensor wrappers or unique
  `ValueId` values; a repeated input still occupies and records each of its semantic positions.
  Internal member edges and an external ADD operand are not additional entries in the current
  representation.
- Count every attempted family at an unclaimed anchor and every longer-to-shorter epilogue form
  tested. Checked arithmetic overflow or inconsistent counting is a malformed-analysis failure,
  never permission to exceed a ceiling.
- Budget exhaustion never changes the already proved split plan and never makes a supported
  partition fail. It only stops further optional recognition.
- The retained-fact ceiling is independently useful at eight, but its over-ceiling case is
  dominated by CPU 0008B: an admitted partition has at most eight nodes and every retained fact
  owns at least one distinct node. Tests must reach eight retained one-node facts and separately
  pin that ownership proof plus `CpuPartitionDagDecomposer.MAX_NODES == 8`; they must not invent a
  supported ninth-fact partition.
- The member, materialized-boundary, and associated-unit ceilings are the exact maxima of the
  closed 0008C matrix, not invitations to synthesize larger forms. Tests must reach the nearest
  real forms and exhaustively prove the owning inventories: six members for Conv1d plus ADD plus
  terminal; ten referenced materialized boundary positions for batch-normalization training's
  five inputs and five outputs; and two associated units for an indivisible ordinary-split anchor
  plus the one CPU 0008B vertically fused pointwise suffix. The boundary maximum must be reached
  end to end with one supported, fully static, resolved-layout `BATCH_NORM_TRAINING` node whose
  five semantic input `ValueId` values are distinct and whose five required output `ValueId`
  values are distinct. A seven-member, eleven-boundary, or three-unit fact would require an
  unlisted form or violate an unchanged parent invariant and is not a valid supported-fallback
  fixture.

  The closed boundary-position inventory is exhaustive: MATMUL records `2 + 1 = 3`; every
  convolution form records at most the convolution anchor's three inputs plus one final result,
  `3 + 1 = 4`; reduction records `1 + 1 = 2`; softmax/log-softmax record `1 + 1 = 2`; Layer Norm
  records at most `3 + 1 = 4`; RMS Norm records at most `2 + 1 = 3`; batch-normalization inference
  records `5 + 1 = 6`; and batch-normalization training records `5 + 5 = 10`. Conv1d composition
  members and suffix members do not add internal-edge access facts under the definition above.
  Therefore no admitted family/form can supply an eleventh recorded position.

### Preparation facts and artifact identity

- Run structural recognition before decomposition so focused code can return the typed MATMUL
  fact, but do not integrate facts into a prepared plan until the complete CPU 0008B baseline has
  succeeded.
- Extend the CPU-private decomposition/preparation result with an immutable ordered list of exact
  recognized facts and their baseline-unit associations. Empty recognition is the default and
  must preserve existing compatibility constructors and plan behavior.
- Validate plan facts against stable member-node ordinals, unit dependencies, recognized family,
  and execution disposition. `EXISTING_SPECIALIZED` must match the exact existing portable IR
  family and epilogue. `ORDINARY_SPLIT` must not alter any unit IR. `UNSUPPORTED_ANCHOR` must not
  enter a successful plan.
- Do not change `CpuGeneratorSchema.CURRENT_VERSION` from 52. Do not add a recognition field to
  `CpuKernelIr`, any family IR, `CpuLoweringFingerprint`, `CpuKernelSpecialization`, or artifact
  envelope. Tests must prove byte-identical structural keys and generated bytes for existing
  Conv2d fused forms, explicit semantic kernels, and ordinary split controls with and without
  recognition enabled.
- No new generated-code or hot-path form is permitted. Therefore this task makes no new
  generated/direct performance claim and runs no redundant timing fork. If implementation needs
  a new emitted form, helper call, generated entry shape, or artifact identity, stop and replan;
  the revised task must require an optimal clean Java oracle, complete Class-File/decompilation
  and hidden-helper inspection, and five-fork `<= 1.15x` generated/direct evidence.

### Documentation and Javadoc

- After executable Java and focused/module validation stabilize, hand the same diff and exact
  evidence to a separate clean documentation-focused agent/thread.
- Finalize meaningful Javadoc for every changed CPU type, record component, constructor, and
  method, including closed variants, ownership, immutability, ordering, budgets, nullability,
  results, failure conditions, and the distinction between recognition and artifact identity.
- Update `docs/backend-guide/cpu-backend.md` with the closed recognition mental model, exact
  family/epilogue matrix, precedence, barriers, hard budgets, ordinary-split fallback, explicit-
  semantic-only rule, and current no-new-execution boundary.
- Review `docs/glossary.md`. Update an existing entry or add one only if the implementation
  introduces a reusable project term or changes an existing term's meaning; otherwise record a
  reasoned no-change conclusion.
- Finalize this task, the CPU master plan, and the roadmap from actual implementation evidence in
  the same overall change.

## Out of scope

- Any new generated kernel, generated entry shape, numerical algorithm, family emitter,
  reference kernel, execution route, capability claim, artifact schema/version, or performance
  claim.
- MATMUL execution, lowering, resources, capability, linear fusion/split, or generated evidence;
  CPU 0008F owns them.
- Conv1d/Conv3d/reduction epilogue fusion or another Conv2d fused form. Recognized forms without
  the exact existing Conv2d artifact remain ordinary split units.
- CPU 0008D profitability ranking, cost estimates, accepted/rejected/selected decision facts,
  candidate comparison, Trace translation, or tuning inspection.
- CPU 0008E external read-boundary materialization variants, multi-input copy selection,
  representation reuse, or candidate re-ranking.
- Pooling, attention, or loss execution or decomposed recognition; CPU 0008G–0008I own those
  executable families.
- Recognition of decomposed softmax, log-softmax, Layer/RMS/batch normalization, attention,
  categorical-cross-entropy, mean-squared error, convolution, or another first-class Model
  semantic.
- A public pattern registry, public fusion API, DSL, callback, matcher plug-in, reflection,
  annotation scanning, string dispatch, generic parameter bag, or public/backend-generic
  recognition contract.
- Compiler canonicalization or pattern recognition, graph rewriting, new Model kind/attribute,
  Runtime graph interpretation or scheduling, Planning kernel knowledge, shared Prepare changes,
  Engine integration, or another module's Java code.
- Dynamic/symbolic Shape execution, unresolved layouts, negative strides, alias speculation,
  representation copying, native-provider packing/reorder, Vector/native routes, tuning-cache
  lookup/mutation, measurement, or Runtime selection.
- Architecture, dependency, module-boundary, Gradle/toolchain, public API, backend-conformance, or
  integration-test changes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): Model semantic ownership; Compiler flat
  graph ownership; concrete-backend lowering/fusion/specialization ownership; staged Prepare;
  Runtime prepared-execution-only boundary; generated-code oracle discipline.
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md): current
  compile/plan/prepare/run navigation.
- [`planning guide`](../../../planning-guide.md): Ready-task completeness, package planning,
  validation tiers, completion evidence, and separate documentation finalization.
- [`CPU master plan`](../master-plan.md): CPU-private package map and ordered 0008B–0008F
  decomposition/recognition/profitability/materialization/MATMUL frontier.
- Complete [`CPU 0008B`](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md):
  deterministic split baseline, units, dependencies, barriers, budgets, resources, and atomic
  sequential execution.
- Complete [`CPU 0008`](0008-portable-grouped-nchw-conv2d-execution-foundation.md) and
  [`CPU 0008A`](0008a-portable-channels-first-dimensional-convolution-closure.md): existing
  Conv2d fused forms, exact Conv1d composition, direct Conv3d, and rank-specific boundaries.
- Complete [`CPU 0007F2`](0007f2-portable-batch-normalization-training-and-statistic-transition-coverage.md):
  explicit-first-class-only normalization and multi-output resource boundary.
- Complete Model tasks
  [`0019`](../../../modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md),
  [`0019D`](../../../modules/model/tasks/0019d-linear-convenience.md),
  [`0018V`](../../../modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md),
  [`0020`](../../../modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md), and
  [`0025H`](../../../modules/model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md).
- Complete Compiler tasks
  [`0003`](../../../modules/compiler/tasks/0003-canonicalization-and-forward-optimization.md),
  [`0005B`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md),
  [`0005D`](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md),
  and [`0006B`](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md).

## Architecture constraints

- Model operation kinds and attributes remain the sole semantic authority. Recognition may
  classify exact existing occurrences but must not synthesize semantic equivalence.
- Compiler retains the immutable flat graph, node/value/publication identities, and canonicalized
  operation order. CPU recognition neither rewrites nor feeds a compiler pass.
- Planning continues to select only CPU ownership. It does not see recognition facts, routes,
  epilogues, unit shapes, or artifact identities.
- CPU analysis owns this private cold recognition and must still declare the exact already
  selected buffers/workspaces before shared assignment. Finalization may realize only the
  already selected existing unit artifacts.
- Runtime receives the same opaque prepared executable(s), resources, access declarations, and
  deterministic unit order. It never sees `Operation`/`CompiledNode`, recognition facts, or a
  graph interpreter in the hot path.
- Any need for a public/shared contract, new operation kind, shared resource, generated form,
  altered artifact identity, new capability, or architecture/dependency change is a stop-and-
  replan condition.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir` — closed immutable recognition facts and
  stable typed structural identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — bounded graph recognition,
  precedence, eligibility, and exact CPU 0008B baseline association.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — immutable cold plan retention and
  validation of recognized facts.

Packages added or changed:

- No package is added. Only the three existing CPU-private packages above may change.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph` — sealed immutable
  typed facts, closed variants/enums, execution disposition, and graph-identity-free structural
  recognition identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSpecializedSubgraphRecognizer` —
  stateless bounded cold recognizer, deterministic precedence, eligibility, budgets, and baseline
  unit association.

Tests mirror production packages. No public test package, registry package, generic pattern
package, or new source set is added.

## Affected files

Expected production/Javadoc paths:

- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuSpecializedSubgraph.java`;
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuSpecializedSubgraphRecognizer.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`; and
- at most one directly affected existing CPU-private `package-info.java` in `internal.ir`,
  `internal.lowering`, or `internal.prepare` if its implemented-boundary Javadoc would otherwise
  become stale.

Expected test paths:

- new focused `CpuSpecializedSubgraphTest` and `CpuSpecializedSubgraphRecognizerTest`;
- existing `CpuPartitionDagDecomposerTest`;
- existing `CpuPartitionPreparationPlanTest` or `CpuPartitionPreparerTest`;
- existing `CpuLoweringFingerprintTest` or the closest current artifact-identity regression
  owner; and
- existing `CpuInternalPackageInventoryTest` only if the new types require inventory changes.

Expected documentation/planning paths during implementation completion:

- `docs/backend-guide/cpu-backend.md`;
- `docs/glossary.md` only when the required review finds a reusable terminology change;
- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

No other module source, architecture, ADR, Gradle, conformance, integration, generated-evidence
resource, or later task-specification path is expected.

## Maximum scope

This task may create or modify at most:

- 6 CPU production/Javadoc paths, including exactly 2 new production types;
- 6 CPU test paths, including at most 2 new test types;
- 5 documentation/planning paths; and
- 17 total paths.

No generated-code source, reference-kernel source, generator schema, retained performance bundle,
or shared-module path may change. If implementation requires another production owner, another
package, a new emitted form, more than 24 recognition attempts, more than 8 facts, more than 10
referenced materialized boundary positions in one fact, more than 2 associated units, or any path
ceiling increase, stop and propose a focused replan rather than silently expanding 0008C.

## Acceptance criteria

- The closed sealed facts represent exactly MATMUL, Conv1d/Conv2d/Conv3d, the ten named floating
  reductions, and the six named explicit semantic kernels, with no generic operation-kind slot,
  string dispatch, registry, DSL, callback, reflection, or mutable collection.
- Recognition is deterministic across repeated runs and input map/hash iteration changes. Exact
  precedence, longest-eligible suffix choice, overlap/ambiguity behavior, fact order, member-node
  order, and baseline-unit associations match this specification.
- MATMUL facts correctly distinguish ordinary versus canonical transposed-weight input and absent
  versus exact rank-one linear bias, but CPU capability/preparation still fails closed before
  declaration/finalization/artifact work.
- Conv2d `ADD` and `ADD -> RELU` facts map only to the existing evidenced fused IR. Every other
  recognized convolution epilogue remains exact ordinary split; direct Conv1d/Conv2d/Conv3d
  behavior remains unchanged.
- Reduction facts cover only the ten named FLOAT32/FLOAT64 anchors with exact current attributes
  and uniform suffix. Masked, target-Shape, arg-extrema, BOOL, integral, BFLOAT16-epilogue, and
  unlisted forms remain ordinary and unrecognized.
- Explicit semantic-kernel facts arise only from the six actual first-class Model kinds and retain
  exact one-unit established lowering. Representative decomposed softmax, log-softmax, Layer/RMS
  normalization, and batch-normalization-like graphs produce no such fact.
- Fan-out, intermediate publication, overlap/alias uncertainty, unresolved layouts, dynamic
  Shapes, non-injective outputs, state/random/multi-output barriers, unsupported pointwise kinds,
  third epilogue operations, and cross-type suffixes reject only recognition and retain the exact
  ordinary baseline where that baseline is supported.
- Reachable seam tests cover 8/over-8 partition nodes, 24/25 attempts, eight retained one-node
  facts, six-member Conv1d plus ADD plus terminal, two epilogue operations plus rejection of a
  literal third operation, ten referenced materialized boundary positions on a supported
  end-to-end batch-normalization-training occurrence, and two associated units on an
  ordinary-split anchor plus suffix. The ten-position case uses one fully static resolved-layout
  node with five distinct ordered inputs and all five distinct ordered outputs—for example, dense
  FLOAT32 input Shape `[2, 3]`, channel axis `1`, four dense `[3]` input vectors, dense output
  Shape `[2, 3]`, and four dense `[3]` statistic outputs—so the CPU 0007F2 domain has `C = 3` and
  `N = 2` and is supported. Every reachable supported over-budget case retains the ordinary
  baseline; the existing over-eight partition rejection remains unchanged.
- Exhaustive invariant tests prove that no admitted graph can supply a ninth retained fact, a
  seven-member fact, an eleventh referenced materialized boundary position, or a third associated
  unit: respectively, CPU 0008B admits at most eight nodes and every fact owns a distinct node;
  the largest anchor has four members and the suffix has at most two; and the closed positional
  boundary inventory is
  MATMUL `2 + 1`, convolution at most `3 + 1`, reduction `1 + 1`, softmax/log-softmax `1 + 1`,
  Layer Norm at most `3 + 1`, RMS Norm at most `2 + 1`, batch-normalization inference `5 + 1`,
  and batch-normalization training `5 + 5`. CPU 0008B retains an admitted anchor as one
  indivisible seed while vertically fusing the eligible pointwise suffix into one unit. These
  dominated over-ceiling cases require proof of the owning stronger invariant, not an artificial
  supported graph, test-only production hook, or unlisted form.
- Recognition budget exhaustion retains already proved facts in stable order and leaves all
  remaining supported work ordinary. Malformed baseline association fails before declarations or
  artifact access.
- Plan validation rejects wrong member ordinals, wrong unit associations, overlapping facts,
  unsupported dispositions in a successful plan, an `EXISTING_SPECIALIZED` fact whose existing
  IR disagrees, and an `ORDINARY_SPLIT` fact that changes unit IR or resource topology.
- With recognition empty, ineligible, or `ORDINARY_SPLIT`, snapshot tests prove exact equality of
  unit members/dependencies, boundary order, buffers, workspaces, carrier/access facts, route,
  specializations, structural keys, and manifest-independent executable order against the CPU
  0008B baseline.
- Existing fused Conv2d, direct Conv1d/Conv2d/Conv3d, reduction, softmax, Layer/RMS, and batch-
  normalization structural keys and regenerated class bytes are byte-identical. Generator schema
  remains 52. No recognition fact enters `CpuKernelIr`, `CpuLoweringFingerprint`,
  `CpuKernelSpecialization`, or the artifact envelope.
- No new generated form exists. Consequently no performance fork is claimed or required. If a
  new form appears, implementation stops and the task is replanned with the repository's optimal-
  clean-Java, complete decompilation/hidden-helper, and five-fork performance gates.
- Existing focused decomposition, family lowering, preparation, finalization, cache identity,
  and CPU module tests remain green.
- No public API, Model, Compiler, Planning, shared Prepare, Runtime, Engine, Backend Contract,
  Config, Trace, provider, dependency, build, architecture, conformance, or integration contract
  changes.
- Production/test types match the package map and every category/path ceiling.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, CPU guide,
  glossary impact, planning evidence/status, and documentation checks in the same overall change.
- CPU 0008C becomes `Review needed` after implementation/evidence pass and remains there until
  the mandatory independent documentation gate passes. On
  completion CPU 0008C becomes `Complete`, CPU 0008D becomes the sole `Ready` CPU row, and no
  0008D or later detailed task specification is created.

## Tests / validation

### Tier 1: focused recognition and unchanged execution contracts

During implementation, run focused tests for:

- sealed fact validation, defensive copies, structural identity, and forbidden identity inputs;
- every admitted family/form/epilogue/disposition;
- precedence, longest suffix, overlap, ambiguity, barriers, budgets, and repeatability;
- exact baseline-unit association and plan validation;
- MATMUL recognition plus unchanged fail-closed capability/preparation;
- Conv2d existing-specialized identity and Conv1d/Conv3d/reduction ordinary split snapshots;
- first-class semantic-kernel positive cases and decomposed-lookalike negative cases; and
- package inventory and artifact-identity exclusion.

Focused, compilation, and earlier full-suite development runs are allowed and must be recorded as
development evidence. After the final executable Java change, designate exactly one subsequent
CPU module suite as the authoritative final run:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

An earlier invocation of the same command is not a product failure and does not invalidate a
later post-stabilization authoritative run; it remains explicitly non-authoritative development
evidence. Executable Java must not change after the designated authoritative run unless that run
is superseded by one new post-change authoritative run.

### Tier 2: generated-code identity evidence

- Regenerate representative existing Conv2d `ADD`/`ADD -> RELU`, direct Conv1d/Conv3d,
  reduction-split, softmax, Layer/RMS, and batch-normalization artifacts through existing tests.
- Compare structural keys, class-byte SHA-256 values, and complete generated member/reference
  scans with the unchanged baseline fixtures. Inspect representative `javap -c -v` output to
  confirm no recognition helper or fact reaches generated hot code.
- Do not run a new performance fork: the task adds no generated or hot-path form and makes no
  performance claim. Record byte-identical reuse as the proportional evidence.

### Tier 3: documentation and repository hygiene

The separate documentation-focused context receives the stabilized diff, final CPU XML, focused
identity evidence, and exact commands. It does not repeat successful Java tests unless executable
behavior changes or it records a concrete stale-evidence risk. After final Javadocs it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates local Markdown links and anchors, required heading order, balanced fences,
terminology, final newlines, trailing whitespace, generated Javadoc pages, exact path/type caps,
schema 52, status synchronization, absence of a 0008D task file, and empty staging.

Repository-wide and architecture tests remain deferred to CPU 0009 or continuous integration
because this task changes one concrete backend's private cold analysis without changing a shared
contract or dependency. Backend-conformance remains CPU 0009, and integration remains Engine
work. A change to any of those facts is a stop condition requiring replanning and proportionate
validation.

## Dependencies

- Complete CPU 0008B: deterministic supported split baseline, one-to-eight-unit topology,
  materialized logical-value edges, per-unit resources, stable dependencies, and atomic sequential
  execution.
- Complete CPU 0008 and CPU 0008A: existing Conv2d fused identity/evidence, exact visible Conv1d
  recognizer, direct Conv3d, and rank-specific convolution semantics.
- Complete CPU 0007A–0007F2: exact reduction and first-class semantic-kernel execution families,
  numerical/resource barriers, and explicit-normalization-only rules.
- Complete Model 0019/0019D, 0018V, 0020, and 0025H plus current operation contracts: exact kinds,
  attributes, Shapes, data types, promotion, composition, and semantics.
- Complete Compiler 0003, 0005B, 0005D, and 0006B: flat canonical graph, validated descriptors,
  publication, and explicit semantic occurrences.
- Existing Planning, Prepare, Runtime, cache, artifact, worker, and generated-code contracts are
  sufficient and unchanged.

All recognition, precedence, budget, fallback, and current execution decisions are fixed. No
architecture or shared-contract blocker is known within this closed static CPU-private scope.

## Follow-up tasks

- CPU 0008D: rank complete already legal fused and split candidates with bounded no-measurement
  profitability heuristics and typed cold decision facts. It must consume 0008C facts without
  redefining recognition.
- CPU 0008E: add bounded single/dual external read-boundary materialization and representation
  reuse variants, then re-rank through 0008D.
- CPU 0008F: implement the complete current MATMUL family and bounded linear epilogues, adding any
  new generated forms, artifact identity, safe split/fusion, oracle, Class-File, and performance
  evidence explicitly.
- CPU 0008G–0008I: pooling, attention, and loss execution at their existing owners.
- CPU 0009: portable generated-coverage, capability, conformance, and integration checkpoint.

No detailed CPU 0008D or later task specification is created by this task.

## Architecture impact

Expected impact: None.

This task exercises existing concrete-backend ownership of private lowering recognition and
retains the existing staged analysis/resource/finalization and Runtime boundaries. If
implementation requires another module, a public/shared recognition type, generated form,
artifact schema change, new execution family, Runtime graph visibility, or architecture rule,
stop and report the exact conflict rather than editing across the boundary.

Reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests remain unchanged
  because ownership and dependency direction do not change.
- Model and public Tensor APIs remain unchanged because all recognized meanings already exist and
  decomposed lookalikes are deliberately not reclassified.
- Compiler APIs/passes remain unchanged because recognition consumes the final flat graph without
  rewriting it.
- Planning, shared Prepare, Runtime, Engine, Backend Contract, Config, Trace, and other backends
  remain unchanged because the facts are CPU-private cold plan metadata.
- Backend conformance and integration tests remain unchanged because no new capability or public
  lifecycle result is promised; CPU 0009 and Engine retain those checkpoints.
- Gradle/build configuration remains unchanged because no module, dependency, source set, or
  toolchain changes.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0008C.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008c-typed-specialized-subgraph-and-epilogue-recognition.md.
Read the task's directly referenced completed CPU/Model/Compiler specifications and the current
CPU decomposer, lowering, IR, preparation, cache-identity source/tests.

Implement exactly the Ready specification. Add typed cold recognition only; do not add a generated
form, MATMUL execution, profitability, multi-input materialization, public registry/DSL, compiler
recognition, Runtime graph interpretation, or later task. Preserve exact CPU 0008B fallback and
schema-52 artifact identity. Enforce the reachable maximum of ten referenced materialized
boundary positions as the five ordered anchor inputs plus five ordered outputs of one supported
first-class `BATCH_NORM_TRAINING` occurrence, and prove an eleventh impossible from the closed
family matrix. Stop and report any architecture, shared-contract, generated-form, or maximum-scope
conflict.

After stable implementation and recorded CPU/artifact-identity validation, hand the same diff and
evidence to a separate clean documentation-focused agent. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs, the CPU
guide, glossary impact, planning evidence/status, and documentation checks in this overall change,
without repeating successful Java tests unless executable behavior changes or a concrete risk is
recorded.

Do not stage, commit, or push. Update this task's local decisions, known limitations, validation
evidence, implementation notes, completion summary, and final status only from actual results.
```

## Local decisions

- 0008C is recognition-only. Adding execution here would either promise MATMUL before 0008F or
  add unevidenced Conv3d/reduction generated forms, violating the ordered family ownership and
  generated-code discipline.
- The uniform epilogue is optional external ADD followed by at most one exact activation/CLAMP.
  This matches the existing Conv2d and planned linear boundary without creating a generic
  pointwise-pattern language.
- MATMUL recognition includes canonical transposed-weight and exact rank-one linear-bias facts,
  but its unsupported disposition cannot enter a prepared plan. This establishes typed semantics
  without making capability untruthful.
- Reduction recognition is limited to ten floating numerical anchors and excludes masked,
  target-Shape, arg-extrema, and BOOL forms. Those excluded forms have distinct input roles,
  binding, index, or result-category contracts and do not justify this epilogue slice.
- First-class softmax and normalization kernels are recognized only by exact kind identity. Shape
  and algebraic similarity never establishes semantic identity.
- Recognition facts remain outside generated artifact identity. A future executable form must
  deliberately add its code-shaping facts and advance compatibility with full generated/direct
  evidence; a cold diagnostic fact must not invalidate byte-identical existing classes.
- The ordinary CPU 0008B plan is both fallback and execution oracle. Recognition cannot remove a
  buffer, unit, workspace, dependency, validation, publication, or operation boundary.
- Planning-correction context `01a03f44-ebfc-7492-b3c6-06f3b70bc9ba` independently corrected the
  materialized-boundary budget from the earlier erroneous value of eight to the proved
  closed-family maximum of ten. The count is exactly `accessFacts.size()`: one position for every
  ordered anchor input followed by every ordered final result, without `ValueId` deduplication.
  Current Model, Compiler, and CPU 0007F2 contracts all fix `BATCH_NORM_TRAINING` at five inputs
  and five outputs, and the recognizer's `build(...)` appends all ten positions. A fully static,
  resolved-layout five-distinct-input/five-distinct-output occurrence reaches ten end to end; the
  exhaustive closed-family matrix proves eleven impossible.
- The preceding correction's other conclusions remain valid: the retained-fact ceiling is eight,
  with a ninth dominated by the unchanged eight-node parent ceiling and distinct-node ownership;
  the associated-unit ceiling is the reachable maximum of two; and six members and two suffix
  operations are reachable exact maxima whose next values require closed-grammar proofs rather
  than synthetic supported fixtures. No family, disposition, recognition topology, fallback,
  generated/hot-path boundary, or public/shared contract changed.
- Exactly one post-stabilization CPU suite is designated authoritative. Earlier focused or full
  development runs remain allowed and recorded; repeating the eventual command during development
  is not itself a product defect or a failed acceptance gate.
- Open questions: None. Exact family inventory, precedence, budgets, barriers, dispositions,
  fallback, identity, validation, and follow-up ownership are fixed; any implementation need
  outside them is a stop-and-replan condition.

## Known limitations

- Recognition does not make MATMUL executable and does not improve runtime performance by itself.
- Conv1d, Conv3d, and reduction epilogues remain materialized split units; only the previously
  implemented exact Conv2d ADD/ADD-RELU form is already specialized.
- BFLOAT16 epilogues remain unrecognized because the current pointwise route does not execute
  BFLOAT16 arithmetic.
- The closed suffix excludes scalar arithmetic, binary multiply, casts, views, multiple ADDs, and
  multiple activations even when a later task might prove one profitable.
- Facts are CPU-private preparation metadata. No public trace, tuning, Engine, or inspection
  surface is added.
- This is not the CPU portable-coverage, conformance, or integration checkpoint; CPU 0009 retains
  that closure.

## Validation evidence

- Clean planning-correction context: `01a03f44-ebfc-7492-b3c6-06f3b70bc9ba`. It read the required
  architecture/planning/task contracts, the complete CPU 0007F2 contract, the current Model and
  Compiler five-input/five-output authorities, and the complete current recognition fact and
  recognizer implementations. It changed no Java, test, generated, guide, glossary, architecture,
  build, conformance, or integration path and ran no Java test.
- Planning-correction validation passed: the ten-position reachable case and complete closed-family
  cardinality matrix are stated consistently in the budget, maximum-scope, acceptance,
  implementation, and completion text; local task links and fragments resolve; canonical heading
  order, balanced fences, final newline, and trailing whitespace are valid; task, master-plan,
  and roadmap statuses keep 0008C as the sole `Ready` CPU task and 0008D as `Draft`; no 0008D task
  file exists; `git diff --check` and `git diff --cached --check` pass; and the staged index is
  empty. The existing implementation worktree remains otherwise untouched.
- The first local heading-order helper used Ruby `Array#filter_map`, which this environment does
  not provide, and failed before evaluating the file. The compatible `map(...).compact` retry
  passed with all 21 headings in canonical order. This was a validator-script compatibility
  failure, not a planning-file, build, or test failure.
- Planning context: `01a03f0f-420b-7172-8faf-ece06da9f2c8`, based on clean `main` at
  `93b3c87a97d60aede5ec9aecfddc6e1394c3b1bd`.
- Targeted local Markdown validation passed for this specification, the CPU master plan, and the
  repository roadmap: every relative target and fragment exists, required task headings are in
  canonical order, all 14 fences are balanced, all three files have final newlines, and
  `git diff --check` reports no whitespace error.
- Status/scope validation confirms this is the only newly created task specification, CPU 0008C
  is the sole `Ready` CPU row, CPU 0008D through CPU 0008I remain `Draft`, and no CPU 0008D task
  file exists.
- Repository-state validation confirms exactly the intended three unstaged planning paths and an
  empty staged diff. No Java, Gradle, Javadoc, generated-code, or performance command is required
  for this planning-only change.
- Initial isolated implementation context `01a03f1b-3640-7122-91e4-7cb8a9ece2a3` added exactly
  the two permitted CPU-private production types. Corrective implementation context
  `01a03f5e-f36a-73a0-b90e-e0ce8bd32d8b` completed the corrected ceilings, reachable/invariant
  seams, exact baseline equality validation, and generated identity evidence without changing
  another module, generated IR, schema, route, capability, public API, or artifact identity.
- Exact supported-baseline facts now retain the existing portable specialization, compute and
  orchestration strategy, extents and range policy, complete materialized boundary bindings,
  carrier forms, affine facts, graph-identity-free materialization parameters, workspace facts,
  direct dependencies, member topology, and the exact zero-base packed geometry consumed by the
  executable. Plan construction independently recomputes these facts and rejects forged ordinary-
  split IR, resources, specialization/scratch, orchestration/range, materialization, and runtime
  topology before the plan can escape to declaration or artifact access.
- The final focused selection passed 36 tests with zero skips, failures, or errors. It includes
  positive one- and two-unit baseline snapshots plus negative forged equality cases. The one new
  plan-validation test was moved into existing `CpuPartitionPreparerTest`, leaving exactly two
  new test types and preserving all coverage.
- Every earlier full run is development evidence only. The one new superseding authoritative
  `./gradlew :backends:cpu:test --rerun-tasks`, executed after the final Java/test change, passed
  101 suites and 522 tests with zero failures, zero errors, and three expected opt-in skips in 12
  seconds; all 22 actionable tasks executed. No executable Java or test changed afterward.
- Retained identity evidence is under `/private/tmp/synaptik-cpu-0008c-evidence.T5vz3e`. Its
  78-entry manifest has SHA-256
  `da42c5829d6ba3eb46f83dcba00aa7060bf82b3e9012dd5b29e0aa72752176ed`. It retains eleven
  representative class files and complete `javap -c -v -p` output for separate Conv2d fused ADD
  and ADD-to-RELU, direct Conv2d/Conv1d reuse, Conv3d, reduction-split, softmax/log-softmax,
  Layer/RMS, and batch-normalization inference/training forms.
- Conv2d ADD and ADD-to-RELU retain structural keys
  `ab352878da4101f85d4e4b836e67230ec87f1b32cc27180a4732246185b15338` and
  `9888ab5a06f14732f631543bc19c6e01129025c0407aaf9789595b884197376f`, and class SHA-256 values
  `e6f54c67c3f3e4c0b8b9a2dc49218bf4b47b381b4db27d68ff3dd5cc5d9174ff` and
  `6d8c69cfd6dee495bf2aa46b0ec05a822dad3f1b563b8387f7afcd42285ae068`. The evidence test proves
  recognition-free portable-IR structural equality and exact generated class-byte identity for
  both, then scans complete member references.
- Binary, complete-disassembly, member, and repository source scans find no recognition fact or
  helper in generated hot code, cache, code generation, executable, or portable-route packages.
  Every stored class checksum verifies. Generator schema remains exactly 52. No timing or
  performance fork ran because recognition adds no generated or hot-path form.
- Clean documentation-focused finalization context
  `01a03f86-c9c9-7ac3-a695-0dd5dbed7628` independently reviewed the architecture, planning,
  documentation profiles, complete task and predecessor contracts, changed production/tests,
  CPU guide, glossary, and retained evidence. It changed Javadocs and Markdown only; executable
  Java and tests remained untouched.
- Final CPU Javadoc generation passed with 11 actionable tasks, two executed and nine up-to-date.
  The only two warnings are the expected Java 26 incubating Vector API module warnings; the new
  recognition contract has no missing-description, record-component, return, or enum-value
  warning.
- Local Markdown links and anchors, all fences, final newlines, canonical 21-heading task order,
  terminology, exact scope, schema 52, status synchronization, absence of a 0008D task file,
  empty staging, `git diff --check`, and `git diff --cached --check` pass. Final scope is exactly
  14 paths: four CPU production/Javadoc paths, six CPU test paths with exactly two new test types,
  and four documentation/planning paths.
- The glossary remains unchanged after reasoned review because specialized-subgraph recognition
  is intentionally CPU-private implementation vocabulary, not a new reusable project term. The
  existing prepare, lowering, kernel IR, artifact, and Runtime entries cover the public
  distinctions used by the guide.
- Java tests and performance forks were not repeated. No executable Java changed after the
  authoritative 101-suite/522-test run, and 0008C adds no generated or hot-path form; the owning
  execution tasks' recorded performance evidence remains controlling.

## Implementation notes

- `CpuSpecializedSubgraph` owns four closed record variants, typed family/form/attribute/access/
  epilogue/disposition projections, defensive snapshots, and graph/artifact-identity-free
  structural equality. MATMUL alone permits `UNSUPPORTED_ANCHOR`; successful-plan validation
  rejects that disposition.
- `CpuSpecializedSubgraphRecognizer` scans stable node order with the fixed family precedence,
  literal suffix vocabulary, barriers, exact baseline-unit association, and corrected hard
  ceilings. Tests reach 24 attempts and block the twenty-fifth, eight retained facts under the
  eight-node parent, six Conv1d-plus-suffix members, two suffix operations, ten distinct supported
  batch-training positions, and two actual dependency-linked units. Closed family cardinalities,
  distinct-node ownership, the four-member-plus-two-suffix grammar, literal third-operation
  rejection, and unchanged 0008B seed/vertical-fusion topology prove the impossible excesses.
- `CpuPartitionPreparer` first obtains the unchanged complete 0008B baseline, recognizes only
  afterward, and attaches facts to the cold plan. The plan validates stable ordering, exact unit
  coverage, non-overlap, unsupported-anchor exclusion, existing Conv2d IR agreement, and exact
  graph-identity-free baseline route/specialization/range/resource/executable topology equality.
- Existing compatibility constructors supply an empty fact list. Recognition is absent from
  `CpuPortableKernelIr`, family IR, `CpuLoweringFingerprint`, `CpuKernelSpecialization`, generated
  entries, resource declarations, finalization, and Runtime execution.

## Completion summary

- Completed changes: Implemented the closed CPU-private recognition fact family, deterministic
  recognizer, exact baseline association, cold plan retention/validation, and focused regressions.
- Files changed or created: Four CPU production/Javadoc paths and six CPU test paths, including
  exactly two new test types; the
  pre-existing task/master/roadmap planning paths remain in the same overall change.
- Tests and validation: The final focused selection passed 36 tests. The one superseding final
  post-stabilization `:backends:cpu:test --rerun-tasks` passed 101 suites/522 tests with three
  expected skips and no failures or errors. Schema, inventory, empty index, whitespace, exact
  baseline equality, corrected reachable budgets, dominated invariants, and scope checks passed.
- Generated-code/artifact identity evidence: Eleven representative existing classes plus complete
  `javap -c -v` scans and companion facts are retained under the checksummed evidence root above;
  recognition/helper references are absent and schema remains 52. No performance fork was run.
- Documentation-agent review: Complete in clean context
  `01a03f86-c9c9-7ac3-a695-0dd5dbed7628`.
- Documentation impact: Finalized the CPU guide's fail-closed recognition boundary and synchronized
  this task, the CPU master plan, and the roadmap. CPU 0008D is the sole Ready row; later tasks
  remain Draft, and no 0008D specification was created.
- Javadoc review: Finalized the four affected production/Javadoc paths and generated CPU Javadoc
  successfully with only the two expected incubating Vector API warnings.
- Glossary impact: No change. Recognition terminology remains CPU-private and the existing
  glossary already defines the reusable lifecycle and artifact terms.
- Architecture/public/shared/build/conformance/integration impact: None. Ownership remains in the
  concrete CPU backend; capability, shared lifecycle/resource contracts, dependencies, Gradle,
  conformance, integration, and other modules are unchanged.
- Unresolved issues: None.
- Required follow-up: CPU 0008D is the sole Ready task; create its detailed specification only in
  its own planning step.

Status: Complete
