# Task 0059: Casts, Layout, and Indexing

## Status

Complete.
Planning was frozen at `c6442b20` and the explicit unfold-padding contract at `645d6e23`.
Implementation landed at `08bf68bb`, focused proof at `a4926b52`, and independent cumulative
Class C review approved every reported P0/P1/P2 remediation.

## Change class

Class C. This is one serial Metal capability, route, native decoder, exact custom-kernel,
preflight, mixed-program, public Engine, package, test, evidence, and documentation change. It
changes no public API and no Model semantics.

## Goal and exact inventory

Complete the structural Metal recipes for the remaining cast, indexing, and layout wires, and
admit production capability only for occurrence domains implemented by an auditable exact custom
route:

- `39 CAST`;
- `69 GATHER_ELEMENTS`, `70 SCATTER_ADD`, `71 GATHER_ND`, `72 SCATTER_ND`, and `73 SELECT`; and
- `74 PAD`, `75 SLICE`, `76 SLICE_UPDATE`, `77 CONCAT`, `78 STACK`, `79 TILE`, `80 FOLD_AXIS`,
  `81 UNFOLD2D`, `82 FOLD2D`, `83 UNFOLD3D`, and `84 FOLD3D`.

Already approved `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, `CONTIGUOUS`,
`GATHER`, `ONE_HOT`, `SCATTER_ELEMENTS`, and `UNFOLD_AXIS` wires `6..19` are regression scope
only. This task must not replace, duplicate, broaden, or reinterpret their recipes or capability.

The starting ledger is exactly `50 true / 65 false`, `62 executable / 53 nonexecutable`, schema
14, identity 15, native ABI 5, and thirteen exports.

## Frozen production gates

### Gate A: proved CAST pairs

`CAST` uses one deterministic integer/raw custom kernel. Capability is occurrence-sensitive and is
true only for the following explicitly proved carrier pairs:

- all six same-type pairs, which are byte-identical copies;
- `BOOL` to each of `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`, using the exact positive
  zero/one carrier encodings;
- each non-BOOL source to `BOOL`, where false is exactly either floating zero sign or integer zero
  and every other raw carrier, including every NaN, is true;
- `INT32 -> INT64` exact sign extension and `INT64 -> INT32` exact low-bit narrowing; and
- `BFLOAT16 -> FLOAT32` exact sixteen-bit left alignment.

Every other source/target pair remains capability-false. In particular, floating narrowing,
`FLOAT64` or `FLOAT32` widening/conversion beyond the named pair, `BFLOAT16 -> FLOAT64`, and
integer-to-floating conversions are blocked until independent per-pair proofs cover rounding,
saturation, signed zero, subnormals, infinities, and NaN sign/quiet/payload rules. No MPSGraph cast
observation, sampled ordinary corpus, tolerance, or host-language cast authorizes them.

The admitted CAST domain permits canonical dense static rank `0..16`, all positive extents for
positive rank, exact input/output Shape equality, and `requiresGrad=false` on both sides. Every
potentially gradient-bearing cast occurrence remains false.

### Gate B: exact read-only indexing

Admit `GATHER_ELEMENTS` and `GATHER_ND` for exact raw payload movement:

- data/output may use any of the six schema carriers;
- index tensors may use `INT32` or `INT64` exactly as Model permits;
- descriptors are canonical dense, fully static, zero-offset, non-view, rank `0..16` where the
  operation contract permits that rank, with positive extents for positive rank;
- every input and output has `requiresGrad=false`;
- normalized axes, shared batch prefixes, tuple depths, Shapes, and selected-axis/result ranks must
  exactly match the Model-derived descriptors; and
- every index value is host-preflighted before command encoding, with no write or partial output on
  the first invalid bound.

The kernel maps one canonical output coordinate to one checked input coordinate and copies exactly
one carrier payload. Scalar GATHER_ND results remain canonical rank zero.
Negative caller axes are already normalized by Model/Compiler; encoded negative or out-of-range
axes are malformed rather than renormalized in native code.

### Gate C: exact copy-only layout

Admit `PAD`, `CONCAT`, `STACK`, and `TILE` for all six carrier representations. Admit
`UNFOLD2D` and `UNFOLD3D` for the Model-legal `FLOAT64`, `FLOAT32`, and `BFLOAT16`
representations. Inputs and outputs are canonical static descriptors with exact Model
Shapes/attributes and no gradients.

- `PAD` copies original payloads and writes the exact same-type raw `ScalarValue` for every padded
  coordinate.
- `CONCAT` and `STACK` preserve input order and use a bounded `1..16` input occurrence domain so
  every buffer is explicitly declared, bound, fingerprinted, and live. Scalar STACK is included;
  scalar CONCAT remains invalid by the Model contract.
- `TILE` implements whole-pattern per-axis repetition, not scalar run repetition. Model-legal
  scalar `PAD` (empty widths) and scalar `TILE` (empty repeats) are exact one-element identities.
- `UNFOLD2D` and `UNFOLD3D` use exact coordinate maps. Direct `Window2dAttrs`/`Window3dAttrs`
  write represented positive zero for out-of-domain samples; explicit `Unfold2dAttrs`/
  `Unfold3dAttrs` write the exact same-type raw `ScalarValue`, including NaN payload/sign,
  infinity, and signed zero. Model excludes BOOL and integer carriers from both window kinds.
  Attribute type/raw bits participate in metadata, fingerprinting, the independent coordinate-map
  oracle, and public raw-value evidence.

Every route writes fresh canonical output storage and preserves all inputs. No raw movement
operation may use a typed MPSGraph selector as its production correctness argument.

### Gate D: explicit blockers

The following remain capability-false under both profiles:

- `SELECT` and `SLICE`, because public Model construction intentionally retains resolved view
  layout and the unchanged schema-14 executable image carries no view strides/offsets; admitting a
  canonical-output raw kernel would violate publication layout and no-alias semantics;
- `SCATTER_ADD` and `SCATTER_ND`, because duplicate membership/order-independent reductions,
  replacement uniqueness, per-type reduction semantics, and gradient-bearing domains are not all
  proved by a single systematic route;
- `SLICE_UPDATE`, because functional replacement plus base/update gradient partition and overlap
  obligations are not proved by the read-only copy kernel;
- `FOLD_AXIS`, `FOLD2D`, and `FOLD3D`, because overlap accumulation is arithmetic and inherits the
  duplicate/reduction result-set and gradient blockers; and
- every otherwise exact new operation occurrence with gradient-bearing metadata, a dynamic or empty
  descriptor, an affine-view input, an unproved CAST pair, unsupported variadic cardinality, or
  malformed/noncanonical attributes.

Do not special-case a nonoverlap geometry to make a blocked reduction kind true. Do not use
atomics, encounter order, tolerance, selector success, or ordinary-value output as proof.

## Structural MPSGraph recipes

Implement native-validated structural recipes for all seventeen wires independently of production
capability:

- direct cast for `CAST`;
- gather-along-axis for `GATHER_ELEMENTS`, direct gather-ND for `GATHER_ND`, and slice-plus-squeeze
  for `SELECT`;
- direct scatter/index selectors where the installed API expresses `SCATTER_ADD` and `SCATTER_ND`;
- direct constant pad, strided slice, slice update, concat, and tile selectors;
- expand-dimensions plus concat for `STACK`;
- the catalog scatter-add composition for `FOLD_AXIS`;
- direct im2col/col2im for `UNFOLD2D`/`FOLD2D`; and
- the catalog's explicit 3D window compositions for `UNFOLD3D`/`FOLD3D`.

When an installed MPSGraph selector has a narrower rank/cardinality contract than Model, the
structural fixture must validate and document that occurrence boundary rather than silently claim
production support. Structural selector presence, executable creation, and bounded output are never
production semantic proof.

After implementation, the structural ledger must be exactly `79 executable / 36 nonexecutable`.
The catalog counts remain `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; only their recipes become
native-executable.

## Custom route and mixed-program design

Add one systematic data-movement custom family plus the bounded proved CAST matrix. The fixed
preparation decision is `CUSTOM_PROGRAM` for every newly admitted occurrence. It is made once,
before resource declaration, and is bound into candidate identity and workload fingerprint.

The custom metadata binds operation wire, source/target carrier widths and types, every Shape,
normalized axis/batch/tuple fact, complete attributes, exact scalar bits, variadic input order and
count, and checked dispatch geometry. Variable inputs are explicit liveness/resource references;
there is no hidden buffer lookup. Mixed programs may combine these nodes with every already approved
custom or nested MPSGraph node. Every intermediate is declared once, allocated once, and consumed
through the existing isolated workspace and direct assigned-output publication.

There is exactly one preparation-fixed route: no fallback, retry, alternate selector after failure,
runtime probing, timing, benchmarking, autotuning, or device-dependent choice.

## Schema, identity, ABI, lifecycle, and errors

Retain schema 14, operation wires `1..115`, type wires `1..6`, attribute wires `0..41`, identity
15, native ABI 5, and exactly thirteen exports because Java already encodes every in-scope node and
attribute. Extending the native decoder to consume existing bytes is not a schema change. If and
only if encoded image bytes genuinely must change, stop and perform one clean version cutover with
no compatibility reader, alias, or dual decoder.

Native parsing independently validates every later attribute kind used here, canonical unused
fields, counts, rank bounds, positive dimensions, signed attribute words, carrier compatibility,
complete Shape equations, topology, and route legality. Malformed schema and raw-ABI tests cover
truncation, wrong kinds/counts/types, invalid axes/batches/tuple depths/steps/windows/scalars, excess
variadic inputs, and illegal custom/MPSGraph route mixing.

Index bounds failures must map to the existing stable range failure without command submission or
partial output. All other invalid occurrence metadata fails before allocation or native creation.
Preserve transactional finalization, reverse/attempt-all rollback, child leases, synchronous runs,
repeated and concurrent execution, close-vs-run ordering, no-alias input/output resources, input
immutability, and direct assigned-output publication.

## Verification and acceptance

1. Freeze this plan before production edits.
2. Add an independent CAST-pair proof/checker that does not share the production transform. It must
   exhaust both BOOL values and all 65,536 BFLOAT16 values for the admitted BFLOAT16 widening and
   BOOL target behavior, partition INT32/INT64 sign-extension/narrowing/zero behavior with exact
   boundary and randomized full-word corroboration, and prove identity pairs by carrier-width copy.
3. Prove raw coordinate maps against an independent CPU/Model oracle across every admitted kind,
   every carrier width, scalar/rank boundaries, negative slice steps, padding, batch/suffix gather,
   window padding/dilation/stride, and variadic placement.
4. Run focused capability/catalog/schema/identity/malformed-native tests and assert the exact final
   ledger. No previously approved operation may change route or domain.
5. Run public no-skip Metal-only Engine smokes for every admitted CAST, indexing, and layout kind,
   including signed zeros, integer extremes, normalized negative caller axes, repeated runs,
   concurrent runs, lifecycle/close ordering, input preservation, output no-alias, and publication
   ownership. Raw native fixtures separately cover all 65,536 BFLOAT16 words, deterministic
   integer full-word corpora, INT64 indices, and bounds-before-write failures.
6. Rebuild/sign/package the native library, verify the exact thirteen exports, run the focused Metal
   package and documentation/Javadoc checks, and do not run a final full repository build.
7. Commit implementation and focused evidence, request an independent cumulative Class C review,
   remediate every P0/P1/P2, rerun affected focused checks, and mark Complete only after approval.

Expected production capability is exactly `59 true / 56 false`: nine newly true operation kinds
(`CAST`, two read-only indexing kinds, and six copy-only layout kinds). The public no-skip gate
keeps view-only `SELECT` and `SLICE` false without weakening their independently executable
structural MPSGraph recipes.

## Completion evidence

- Production capability is exactly `59 true / 56 false`: wires `39`, `69`, `71`, `74`, `77`,
  `78`, `79`, `81`, and `83` are newly true under the frozen occurrence gates; the other eight
  Task-0059 wires remain false for the recorded blockers.
- Structural execution is exactly `79 executable / 36 nonexecutable`; catalog state is unchanged at
  `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`, while the custom catalog is
  `38 AVAILABLE / 77 PENDING / 0 UNAVAILABLE_WITH_PROOF`.
- Focused native and Java proof passed for all nineteen CAST pairs, every BFLOAT16 word, integer
  boundary/full-word corpora, all carrier-width raw movement, both index widths, bounds-before-write,
  internal produced indices, scalar policy, rank-sixteen attributes, signed extreme slice strides,
  ceil-window tails, fold masking/wide geometry, mixed custom/nested programs, schema/malformed
  controls, candidate identity, prepared resources, and exact ledgers.
- No-skip public Metal Engine proof passed for every admitted kind, scalar results, exact
  `GATHER_ELEMENTS`/`GATHER_ND` diagnostics, repeated and independent sessions, concurrent runs,
  direct publication ownership, input preservation, no-alias results, and close ordering.
- The rebuilt dylib was ad-hoc signed, locally packaged, and independently verified with ABI 5,
  schema 14, identity 15, and exactly thirteen exports. The opt-in Gradle package verifier and
  Metal Javadocs passed. Per plan, no final full repository build was run.
- Independent cumulative Class C review approved the final source after confirming remediation of
  staged internal-index validation, all-six-carrier feeds and splats, exact diagnostic replay,
  nested MPS compaction, structural type/shape parity, zero-prefix crop, rank-sixteen attributes,
  overflow-safe slices, ceil-tail windows, and masked 64-bit Fold3D depth geometry.
