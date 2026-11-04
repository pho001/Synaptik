# Task 0027: Explicit Numerical-Profile Semantic Authority

**Editorial reading key (current context, not a status update).**

This historical task establishes Model’s numerical meaning, not backend capability or a
configuration API. `STRICT_IEEE` retains each operation’s existing contract; `ACCELERATOR` permits
only the listed operation-specific alternatives, not generic tolerance. DAZ means treating a
permitted subnormal input as same-signed zero; FTZ means flushing a permitted subnormal arithmetic
result to zero. For example, accelerator arithmetic may flush a tiny result, but a selected `WHERE`
value gains no such permission. Scan traversal and exclusive identities remain fixed; NaN exclusions
still apply.

The profile names belong to this historical decision. The current [data-type and arithmetic
boundaries](../../../../reference/tensor/data-types.md#data-types) explain working formats and exact
movement without restoring graph-selectable profiles. A caller supplies a kind and compatible
immutable attributes to form an `Operation` description; input values belong to particular
occurrences, not that reusable descriptor. Signatures count input/output positions, not elements,
and establish neither numerical validity nor backend support. Model defines meaning; execution
providers realize supported routes without redefining it. This semantic responsibility is not buffer
ownership, and describing an operation does not execute it.

## Status

Complete

Frontier verification established [Metal](../../../../glossary.md#term-metal) 0014 as Complete at implementation [commit](../../../../glossary.md#term-commit) `01e81be2`
after its required [validation](../../../../glossary.md#term-validation) and independent Class C `APPROVE` with zero findings. [Model](../../../../glossary.md#term-model) 0026
remained an independent Draft [FLOAT16](../../../../glossary.md#term-float16) branch, and no other numerical-profile task was [Ready](../../../../glossary.md#term-ready) or in
progress.

Implementation commit `ff86a302` passed worker validation. Independent validation then passed
1,114 Model tests and nine architecture tests with zero failures and zero skips, and the mandatory
independent Class C review returned `APPROVE` with zero findings.

## Change class

Class C — this establishes cross-[module](../../../../glossary.md#term-module) numerical-semantic authority and the bounded result sets
that later Config, [Planning](../../../../glossary.md#term-planning), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), and [Metal](../../../../glossary.md#term-metal) work must transport or
realize. It changes no executable [backend capability](../../../../glossary.md#backend-capability) or public configuration [API](../../../../glossary.md#term-application-programming-interface-api).

## Goal

Make [Model](../../../../glossary.md#term-model) the unambiguous [owner](../../../../glossary.md#term-owner) of two [graph](../../../../glossary.md#graph) numerical profiles. `STRICT_IEEE` preserves each
[operation](../../../../glossary.md#operation)'s current contract; it is not a universal bitwise, [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero), Java `strictfp`, or
fixed-instruction promise. [`ACCELERATOR`](../../../../glossary.md#term-accelerator) is an opt-in permission for only the operation-specific
alternative results listed by this task. It is never a generic [tolerance](../../../../glossary.md#term-tolerance), fallback, or permission
for gross errors.

## [Scope](../../../../glossary.md#term-scope)

- Update the root authority index and global invariants so numerical-profile semantics have one
  owner and later lifecycle layers may transport, retain, and realize one selected graph profile
  without reinterpreting it.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lifecycle glossary](../../../../glossary.md#lifecycle); [graph glossary](../../../../glossary.md#graph))</span>
- Update the foundational scoped contract so Model owns profile-indexed result sets, Config will
  own only the declarative selector, Planning will ask profile-qualified capability questions, and
  concrete backends will later realize an allowed result during Prepare.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [backends glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Add ADR 0019 recording the graph-wide choice, rejected fast-math boolean/tolerance/per-backend
  alternatives, bounded transformations, cold lifecycle placement, cache identity requirement,
  and trace deferral.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [boolean glossary](../../../../glossary.md#data-type--datatype); [tolerance glossary](../../../../glossary.md#term-tolerance); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [trace glossary](../../../../glossary.md#trace))</span>
- Add one authoritative operation-family table and align affected Model operation Javadocs. The
  table is limited to current <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>[`FLOAT32`](../../../../glossary.md#term-float32); all unlisted types and operations keep current semantics.
- Update the focused module-boundary explanation and targeted glossary/API wording needed to keep
  current-versus-planned status explicit.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

### Required profile meanings

| Profile | Required meaning |
|---|---|
| `STRICT_IEEE` | The exact current per-[operation](../../../../glossary.md#operation) [Model](../../../../glossary.md#term-model) contract, including every current family-specific [rounding](../../../../glossary.md#term-rounding), approximation, [NaN](../../../../glossary.md#term-nan)-payload, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), accumulation, [reassociation](../../../../glossary.md#term-reassociation), and [FMA](../../../../glossary.md#term-fused-multiply-add-fma) promise or freedom. |
| [`ACCELERATOR`](../../../../glossary.md#term-accelerator) | A superset of the strict allowed-result set formed only by the row-specific transformations below. A [backend](../../../../glossary.md#backend) may always choose a strict result. |

### Required bounded [`ACCELERATOR`](../../../../glossary.md#term-accelerator) table

| [Operation](../../../../glossary.md#operation) family | Additional allowed results |
|---|---|
| [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../../../../glossary.md#term-scalar) `ADD`, `SUB`, `MUL`, `DIV` | A declared arithmetic [`FLOAT32`](../../../../glossary.md#term-float32) [subnormal](../../../../glossary.md#term-subnormal) input may be interpreted as same-[signed zero](../../../../glossary.md#term-signed-zero) ([DAZ](../../../../glossary.md#term-daz)); a [finite](../../../../glossary.md#term-finite) subnormal result may be flushed to zero ([FTZ](../../../../glossary.md#term-ftz)). An FTZ zero may use either sign. An exact-zero `ADD`/`SUB` result may use either zero sign. `MUL`/`DIV` otherwise retain the sign required by the selected operands. |
| `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `SUM_TO_SHAPE` | Use all and only the declared terms for each output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) in a `FLOAT32` binary tree with per-step [rounding](../../../../glossary.md#term-rounding) and the row's DAZ/FTZ rules. Reordering or [reassociation](../../../../glossary.md#term-reassociation) may change the tree but may not drop, duplicate, or invent a term. `MEAN` divides the selected sum by the declared positive count. |
| [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul), [`CONV2D`](../../../../glossary.md#convolution--conv2d), [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) | Retain the current family-owned reassociation/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) permission and additionally allow row-scoped DAZ/FTZ. FMA contracts only a corresponding multiply/add in the declared [contraction](../../../../glossary.md#contraction-dimension); it does not fuse arbitrary [graph](../../../../glossary.md#graph) [nodes](../../../../glossary.md#node) or erase an observable intermediate value. |
| Floating comparisons | Compare row-scoped DAZ-normalized operands; retain the current [NaN](../../../../glossary.md#term-nan) and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) truth tables and [canonical](../../../../glossary.md#term-canonical-form) [`BOOL`](../../../../glossary.md#data-type--datatype) result. |
| Binary/scalar/reduction `MIN` and `MAX` | NaN still propagates. Opposite-zero ties may return either zero sign. Values equal after DAZ [normalization](../../../../glossary.md#term-normalization) may return either original operand bit pattern; unequal normalized values retain numeric ordering. |
| `CUM_SUM`, `CUM_PROD` | Preserve [axis](../../../../glossary.md#term-axis), direction, traversal, inclusive/exclusive placement, and exact exclusive `+0`/`+1` [identities](../../../../glossary.md#term-identity-meanings). Only arithmetic steps may use row-scoped DAZ/FTZ and either sign for an FTZ zero. |
| Affine/[layout](../../../../glossary.md#layout) movement, [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), `WHERE`, classification, [Boolean](../../../../glossary.md#data-type--datatype) logic, casts, indexing, ordering, and arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) | No relaxation. Preserve their current bit, conversion, truth, selected-value, index, and tie contracts. |
| Unlisted [unary](../../../../glossary.md#term-unary)/transcendental, normalization, [loss](../../../../glossary.md#loss), attention, [pooling](../../../../glossary.md#term-pooling), scatter/fold, random, and recurrent families | No relaxation in this task. A later architecture update must give a named operation its own finite-domain and special-value envelope before a [backend](../../../../glossary.md#backend) may use `ACCELERATOR` to widen capability. |

For a fixed permitted [evaluation](../../../../glossary.md#term-evaluation), NaN sign, payload, and quiet/signaling representation may vary
only where the current operation has no stronger promise. NaN may not become an ordinary value and
an ordinary value may not become NaN except through a genuine [IEEE](../../../../glossary.md#term-ieee) exceptional path created by a
listed DAZ, reassociation, or FMA choice. Transcendentals receive no generic [epsilon](../../../../glossary.md#epsilon) or [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) budget.

### Mandatory gross-error exclusions

- No term dropping, reciprocal substitution, algebraic identity rewrite, arbitrary reduced
  precision, cross-node contraction, or tolerance-based acceptance.<span data-glossary-navigation="true"> ([algebraic identity glossary](../../../../glossary.md#term-algebraic-identity))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [precision glossary](../../../../glossary.md#term-precision); [node glossary](../../../../glossary.md#node); [contraction glossary](../../../../glossary.md#contraction-dimension); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- `maxFinite / maxFinite -> 0` and `+infinity / maxFinite -> NaN` remain invalid.
- `RELU(NaN) -> +0` and `TANH(NaN) -> +1` remain invalid; zero-sign freedom never changes NaN
  classification.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>
- `CAST`, affine/[layout](../../../../glossary.md#layout) movement, [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), and selected `WHERE` values do not acquire DAZ/FTZ
  or NaN-payload freedom from surrounding arithmetic.<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
- [`FLOAT64`](../../../../glossary.md#term-float64), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), future [`FLOAT16`](../../../../glossary.md#term-float16), integral values, and [`BOOL`](../../../../glossary.md#data-type--datatype) receive no relaxed arithmetic
  meaning. Model 0026 remains the independent owner of future IEEE FLOAT16 semantics.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [IEEE glossary](../../../../glossary.md#term-ieee))</span>
- This task authorizes no Config enum, capability row, backend route, native schema, cache codec,
  Engine builder method, Runtime field, or trace payload.<span data-glossary-navigation="true"> ([backend route glossary](../../../../glossary.md#backend-route); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [codec glossary](../../../../glossary.md#term-codec); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [trace payload glossary](../../../../glossary.md#trace-payload--tracepayload))</span>

## Non-goals

Do not edit production behavior, tests, [Gradle](../../../../glossary.md#term-gradle) dependencies, [Planning](../../../../glossary.md#term-planning)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) APIs,
[backend capability](../../../../glossary.md#backend-capability), route/candidate code, [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), [Runtime](../../../../glossary.md#term-runtime), or [Trace](../../../../glossary.md#trace). Do not retroactively mark
[Metal](../../../../glossary.md#term-metal) 0006–0007 or 0009–0013 unblocked. Their strict evidence remains valid until a later profile-
specific successor passes a fresh bounded [oracle](../../../../glossary.md#term-test-oracle).

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>, and [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates) — establish one [owner](../../../../glossary.md#term-owner) before cross-[scope](../../../../glossary.md#term-scope) implementation and perform the coordinated architecture update.
- [Foundational modules — Scope](../../../../architecture/contracts/foundational-modules.md#scope)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>, [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel), [`modules/config`](../../../../architecture/contracts/foundational-modules.md#modulesconfig), and [`modules/planning`](../../../../architecture/contracts/foundational-modules.md#modulesplanning) — [Model](../../../../glossary.md#term-model) owns semantics; Config and [Planning](../../../../glossary.md#term-planning) must not acquire semantic or route logic.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend))</span> — later [backends](../../../../glossary.md#backend) own truthful profile-qualified capability and realization, not this task.

If one authoritative owner or a bounded [operation](../../../../glossary.md#operation) row cannot be stated without overlapping scoped
authority, stop and report the ambiguity; do not distribute the semantic rules across later tasks.

## Dependencies and integration

- Depends on: completed Model operation-family semantics through 0025L; completed Metal 0014 and
  retained blocker evidence from Metal 0005–0007 and 0009–0013; explicit user acceptance of
  numerical-profile Variant B.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- Conflicts with: any concurrent edit to `ARCHITECTURE.md`, `foundational-modules.md`, affected
  operation-family Javadocs, ADR numbering, or numerical-profile planning rows.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None.
- Common base revision: `01e81be217fb89bdaeedb60a939bb8a734669a24`.
- Integration order: first; before Config 0006, [Engine](../../../../glossary.md#term-engine) 0018, [CPU](../../../../glossary.md#term-cpu) 0017, or [Metal](../../../../glossary.md#term-metal) 0015.
- Integration [validation](../../../../glossary.md#term-validation): numerical-profile semantic-authority [checkpoint](../../../../glossary.md#term-checkpoint).
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner.

## Files and symbols

- `ARCHITECTURE.md` — numerical-profile authority index and global [lifecycle](../../../../glossary.md#lifecycle) [invariant](../../../../glossary.md#term-invariant).
- `docs/architecture/contracts/foundational-modules.md` — sole normative result-set table and
  Model/Config/Planning ownership split.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `docs/design/decisions/0019-explicit-numerical-profiles.md` — accepted decision and alternatives.
- `docs/architecture/module-boundaries.md`, `docs/api/tensor-api.md`, `docs/glossary.md` — focused
  explanation and current-versus-planned terminology.
- `BinaryArithmeticKind`, `ScalarElementwiseKind`, `BinaryComparisonKind`,
  `AggregateReductionKind`, `CumulativeScanKind`, `MatmulKind`, `Conv2dKind`, and `Conv3dKind` —
  profile-indexed Javadocs consistent with the single authoritative table.
- [Model](../../../../glossary.md#term-model)/config/[engine](../../../../glossary.md#term-engine)/[CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) master plans and `docs/planning/roadmap.md` — status only; retain
  exactly one Ready frontier and do not detail dependent Draft tasks.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Acceptance criteria

- Root and foundational scopes name one unambiguous semantic owner; no other scoped contract gains
  a competing result-set definition.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- The complete table and exclusions above are normative, internally consistent, and reflected in
  every affected family Javadoc without implying executable evaluation in Model.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [Javadoc glossary](../../../../glossary.md#term-javadoc); [evaluation glossary](../../../../glossary.md#term-evaluation); [Model glossary](../../../../glossary.md#term-model))</span>
- Strict behavior is described as current per-[operation](../../../../glossary.md#operation) behavior, not universal bitwise strictness.
- Accelerator behavior is opt-in, operation-specific, and a superset permission; the four named
  gross errors remain excluded.<span data-glossary-navigation="true"> ([Accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- No public/configuration API, implementation behavior, backend support, Runtime policy, or trace
  surface changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [trace glossary](../../../../glossary.md#trace))</span>
- Config 0006, Engine 0018, CPU 0017, and Metal 0015 remain Draft; Model 0026 remains independent
  Draft; this task remains the sole Ready frontier until completed and independently reviewed.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [Ready glossary](../../../../glossary.md#term-ready))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Also inspect generated Javadocs for every affected enum, validate local links/anchors and balanced
fences, and confirm the diff contains no executable Java statement, test, [Gradle](../../../../glossary.md#term-gradle), [backend](../../../../glossary.md#backend), native,
[Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Compiler](../../../../glossary.md#term-compiler), [Engine](../../../../glossary.md#term-engine), Config [API](../../../../glossary.md#term-application-programming-interface-api), or [Trace](../../../../glossary.md#trace) change.

Integration/repository validation: [run](../../../../glossary.md#term-run) the repository architecture/documentation [checkpoint](../../../../glossary.md#term-checkpoint) once
after the independent Class C review; no backend/[conformance](../../../../glossary.md#term-conformance)/integration execution is required
because this task advertises no capability.

## Follow-up

- Config 0006 adds only the two-value declarative [identity](../../../../glossary.md#term-identity-meanings) after this contract is Complete.
- [Engine](../../../../glossary.md#term-engine) 0018 performs the [atomic](../../../../glossary.md#resource-transaction-and-rollback) Config-to-[Planning](../../../../glossary.md#term-planning)-to-[Compiler](../../../../glossary.md#term-compiler)-to-[Prepare](../../../../glossary.md#prepare)-to-Engine propagation.
- [CPU](../../../../glossary.md#term-cpu) 0017 and [Metal](../../../../glossary.md#term-metal) 0015 remain separate profile-realization tasks after the spine is Complete.

## Documentation and review impact

- This is an architecture, public numerical-contract, terminology, and Javadoc change. A separate
  clean documentation/review context is mandatory.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Independent Class C review must inspect authority, result-set boundedness, gross-error exclusions,
  unchanged strict semantics, current-versus-planned wording, and absence of implementation drift.

## Result

Implemented the coordinated root/foundational authority update, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0019, focused architecture/[API](../../../../glossary.md#term-application-programming-interface-api)/
glossary wording, and all eight affected [operation](../../../../glossary.md#operation)-family Javadocs. The foundational contract now
contains the sole [normative](../../../../glossary.md#term-normative) profile table and mandatory gross-error exclusions; no executable
behavior, public/config API, shared record, [backend](../../../../glossary.md#backend), native, [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Compiler](../../../../glossary.md#term-compiler), [Engine](../../../../glossary.md#term-engine),
[Trace](../../../../glossary.md#trace), [Gradle](../../../../glossary.md#term-gradle), or test file changed.

Worker [validation](../../../../glossary.md#term-validation) passed `:modules:model:test`, `:modules:model:javadoc`, and
`:testing:architecture-tests:test`; generated pages for every affected enum contain the profile
contract and sole-table link; changed-document links, anchors, fences, newlines, and whitespace
passed validation; and `git diff --check` passed. The repository validator also reported the
pre-existing duplicate generic section anchors in `docs/api/tensor-api.md`; this change added no
heading there and introduced no new duplicate.

Independent validation passed 1,114 [Model](../../../../glossary.md#term-model) tests and nine architecture tests with zero failures and
zero skips. Mandatory independent Class C review returned `APPROVE` with zero findings. Model 0027
is Complete at implementation [commit](../../../../glossary.md#term-commit) `ff86a302`; dependent Config 0006 is the next [authorized
frontier](../../../../glossary.md#term-authorized-frontier).
