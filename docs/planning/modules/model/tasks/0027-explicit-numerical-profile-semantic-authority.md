# Task 0027: Explicit Numerical-Profile Semantic Authority

## Status

Ready

Frontier verification: Metal 0014 is Complete at implementation commit `01e81be2` after its
required validation and independent Class C `APPROVE` with zero findings. Model 0026 remains an
independent Draft FLOAT16 branch. No other numerical-profile task is Ready or in progress.

## Change class

Class C — this establishes cross-module numerical-semantic authority and the bounded result sets
that later Config, Planning, Compiler, Prepare, Engine, CPU, and Metal work must transport or
realize. It changes no executable backend capability or public configuration API.

## Goal

Make Model the unambiguous owner of two graph numerical profiles. `STRICT_IEEE` preserves each
operation's current contract; it is not a universal bitwise, correctly rounded, Java `strictfp`, or
fixed-instruction promise. `ACCELERATOR` is an opt-in permission for only the operation-specific
alternative results listed by this task. It is never a generic tolerance, fallback, or permission
for gross errors.

## Scope

- Update the root authority index and global invariants so numerical-profile semantics have one
  owner and later lifecycle layers may transport, retain, and realize one selected graph profile
  without reinterpreting it.
- Update the foundational scoped contract so Model owns profile-indexed result sets, Config will
  own only the declarative selector, Planning will ask profile-qualified capability questions, and
  concrete backends will later realize an allowed result during Prepare.
- Add ADR 0019 recording the graph-wide choice, rejected fast-math boolean/tolerance/per-backend
  alternatives, bounded transformations, cold lifecycle placement, cache identity requirement,
  and trace deferral.
- Add one authoritative operation-family table and align affected Model operation Javadocs. The
  table is limited to current `FLOAT32`; all unlisted types and operations keep current semantics.
- Update the focused module-boundary explanation and targeted glossary/API wording needed to keep
  current-versus-planned status explicit.

### Required profile meanings

| Profile | Required meaning |
|---|---|
| `STRICT_IEEE` | The exact current per-operation Model contract, including every current family-specific rounding, approximation, NaN-payload, signed-zero, accumulation, reassociation, and FMA promise or freedom. |
| `ACCELERATOR` | A superset of the strict allowed-result set formed only by the row-specific transformations below. A backend may always choose a strict result. |

### Required bounded `ACCELERATOR` table

| Operation family | Additional allowed results |
|---|---|
| Binary/scalar `ADD`, `SUB`, `MUL`, `DIV` | A declared arithmetic `FLOAT32` subnormal input may be interpreted as same-signed zero (DAZ); a finite subnormal result may be flushed to zero (FTZ). An FTZ zero may use either sign. An exact-zero `ADD`/`SUB` result may use either zero sign. `MUL`/`DIV` otherwise retain the sign required by the selected operands. |
| `SUM`, `MEAN`, `SUM_TO_SHAPE` | Use all and only the declared terms for each output coordinate in a `FLOAT32` binary tree with per-step rounding and the row's DAZ/FTZ rules. Reordering or reassociation may change the tree but may not drop, duplicate, or invent a term. `MEAN` divides the selected sum by the declared positive count. |
| `MATMUL`, `CONV2D`, `CONV3D` | Retain the current family-owned reassociation/FMA permission and additionally allow row-scoped DAZ/FTZ. FMA contracts only a corresponding multiply/add in the declared contraction; it does not fuse arbitrary graph nodes or erase an observable intermediate value. |
| Floating comparisons | Compare row-scoped DAZ-normalized operands; retain the current NaN and signed-zero truth tables and canonical `BOOL` result. |
| Binary/scalar/reduction `MIN` and `MAX` | NaN still propagates. Opposite-zero ties may return either zero sign. Values equal after DAZ normalization may return either original operand bit pattern; unequal normalized values retain numeric ordering. |
| `CUM_SUM`, `CUM_PROD` | Preserve axis, direction, traversal, inclusive/exclusive placement, and exact exclusive `+0`/`+1` identities. Only arithmetic steps may use row-scoped DAZ/FTZ and either sign for an FTZ zero. |
| Affine/layout movement, `CONTIGUOUS`, `WHERE`, classification, Boolean logic, casts, indexing, ordering, and arg-extrema | No relaxation. Preserve their current bit, conversion, truth, selected-value, index, and tie contracts. |
| Unlisted unary/transcendental, normalization, loss, attention, pooling, scatter/fold, random, and recurrent families | No relaxation in this task. A later architecture update must give a named operation its own finite-domain and special-value envelope before a backend may use `ACCELERATOR` to widen capability. |

For a fixed permitted evaluation, NaN sign, payload, and quiet/signaling representation may vary
only where the current operation has no stronger promise. NaN may not become an ordinary value and
an ordinary value may not become NaN except through a genuine IEEE exceptional path created by a
listed DAZ, reassociation, or FMA choice. Transcendentals receive no generic epsilon or ULP budget.

### Mandatory gross-error exclusions

- No term dropping, reciprocal substitution, algebraic identity rewrite, arbitrary reduced
  precision, cross-node contraction, or tolerance-based acceptance.
- `maxFinite / maxFinite -> 0` and `+infinity / maxFinite -> NaN` remain invalid.
- `RELU(NaN) -> +0` and `TANH(NaN) -> +1` remain invalid; zero-sign freedom never changes NaN
  classification.
- `CAST`, affine/layout movement, `CONTIGUOUS`, and selected `WHERE` values do not acquire DAZ/FTZ
  or NaN-payload freedom from surrounding arithmetic.
- `FLOAT64`, `BFLOAT16`, future `FLOAT16`, integral values, and `BOOL` receive no relaxed arithmetic
  meaning. Model 0026 remains the independent owner of future IEEE FLOAT16 semantics.
- This task authorizes no Config enum, capability row, backend route, native schema, cache codec,
  Engine builder method, Runtime field, or trace payload.

## Non-goals

Do not edit production behavior, tests, Gradle dependencies, Planning/Compiler/Prepare/Engine APIs,
backend capability, route/candidate code, native code, Runtime, or Trace. Do not retroactively mark
Metal 0006–0007 or 0009–0013 unblocked. Their strict evidence remains valid until a later profile-
specific successor passes a fresh bounded oracle.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), and [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates) — establish one owner before cross-scope implementation and perform the coordinated architecture update.
- [Foundational modules — Scope](../../../../architecture/contracts/foundational-modules.md#scope), [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel), [`modules/config`](../../../../architecture/contracts/foundational-modules.md#modulesconfig), and [`modules/planning`](../../../../architecture/contracts/foundational-modules.md#modulesplanning) — Model owns semantics; Config and Planning must not acquire semantic or route logic.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules) — later backends own truthful profile-qualified capability and realization, not this task.

If one authoritative owner or a bounded operation row cannot be stated without overlapping scoped
authority, stop and report the ambiguity; do not distribute the semantic rules across later tasks.

## Dependencies and integration

- Depends on: completed Model operation-family semantics through 0025L; completed Metal 0014 and
  retained blocker evidence from Metal 0005–0007 and 0009–0013; explicit user acceptance of
  numerical-profile Variant B.
- Conflicts with: any concurrent edit to `ARCHITECTURE.md`, `foundational-modules.md`, affected
  operation-family Javadocs, ADR numbering, or numerical-profile planning rows.
- Parallel group: None.
- Common base revision: `01e81be217fb89bdaeedb60a939bb8a734669a24`.
- Integration order: first; before Config 0006, Engine 0018, CPU 0017, or Metal 0015.
- Integration validation: numerical-profile semantic-authority checkpoint.
- Shared-document integration owner: Main planner.

## Files and symbols

- `ARCHITECTURE.md` — numerical-profile authority index and global lifecycle invariant.
- `docs/architecture/contracts/foundational-modules.md` — sole normative result-set table and
  Model/Config/Planning ownership split.
- `docs/design/decisions/0019-explicit-numerical-profiles.md` — accepted decision and alternatives.
- `docs/architecture/module-boundaries.md`, `docs/api/tensor-api.md`, `docs/glossary.md` — focused
  explanation and current-versus-planned terminology.
- `BinaryArithmeticKind`, `ScalarElementwiseKind`, `BinaryComparisonKind`,
  `AggregateReductionKind`, `CumulativeScanKind`, `MatmulKind`, `Conv2dKind`, and `Conv3dKind` —
  profile-indexed Javadocs consistent with the single authoritative table.
- Model/config/engine/CPU/Metal master plans and `docs/planning/roadmap.md` — status only; retain
  exactly one Ready frontier and do not detail dependent Draft tasks.

## Acceptance criteria

- Root and foundational scopes name one unambiguous semantic owner; no other scoped contract gains
  a competing result-set definition.
- The complete table and exclusions above are normative, internally consistent, and reflected in
  every affected family Javadoc without implying executable evaluation in Model.
- Strict behavior is described as current per-operation behavior, not universal bitwise strictness.
- Accelerator behavior is opt-in, operation-specific, and a superset permission; the four named
  gross errors remain excluded.
- No public/configuration API, implementation behavior, backend support, Runtime policy, or trace
  surface changes.
- Config 0006, Engine 0018, CPU 0017, and Metal 0015 remain Draft; Model 0026 remains independent
  Draft; this task remains the sole Ready frontier until completed and independently reviewed.

## Validation

Worker validation:

```bash
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Also inspect generated Javadocs for every affected enum, validate local links/anchors and balanced
fences, and confirm the diff contains no executable Java statement, test, Gradle, backend, native,
Runtime, Prepare, Compiler, Engine, Config API, or Trace change.

Integration/repository validation: run the repository architecture/documentation checkpoint once
after the independent Class C review; no backend/conformance/integration execution is required
because this task advertises no capability.

## Follow-up

- Config 0006 adds only the two-value declarative identity after this contract is Complete.
- Engine 0018 performs the atomic Config-to-Planning-to-Compiler-to-Prepare-to-Engine propagation.
- CPU 0017 and Metal 0015 remain separate profile-realization tasks after the spine is Complete.

## Documentation and review impact

- This is an architecture, public numerical-contract, terminology, and Javadoc change. A separate
  clean documentation/review context is mandatory.
- Independent Class C review must inspect authority, result-set boundedness, gross-error exclusions,
  unchanged strict semantics, current-versus-planned wording, and absence of implementation drift.

## Result

Empty until execution.
