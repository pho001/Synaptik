# Task 0059: Casts, Layout, and Indexing

## Status

Ready and active from clean revision `67d68071`.

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

Admit `GATHER_ELEMENTS`, `GATHER_ND`, and `SELECT` for exact raw payload movement:

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
one carrier payload. Scalar SELECT and scalar GATHER_ND results remain canonical rank zero.
Negative caller axes are already normalized by Model/Compiler; encoded negative or out-of-range
axes are malformed rather than renormalized in native code.

### Gate C: exact copy-only layout

Admit `PAD`, `SLICE`, `CONCAT`, `STACK`, `TILE`, `UNFOLD2D`, and `UNFOLD3D` for all six carrier
representations, canonical static descriptors, exact Model Shapes/attributes, and no-gradient
inputs/outputs.

- `PAD` copies original payloads and writes the exact same-type raw `ScalarValue` for every padded
  coordinate.
- `SLICE` supports normalized finite signed steps, including reversal, and copies selected payloads
  bit-for-bit. `CropToShapeAttrs` is admitted only after it is lowered to one exact concrete region;
  unresolved/dynamic crop occurrences remain false.
- `CONCAT` and `STACK` preserve input order and use a bounded `1..16` input occurrence domain so
  every buffer is explicitly declared, bound, fingerprinted, and live. Scalar STACK is included;
  scalar CONCAT remains invalid by the Model contract.
- `TILE` implements whole-pattern per-axis repetition, not scalar run repetition.
- `UNFOLD2D` and `UNFOLD3D` use exact coordinate maps. Direct `Window2dAttrs`/`Window3dAttrs`
  write represented positive zero for out-of-domain samples; explicit `Unfold2dAttrs`/
  `Unfold3dAttrs` write the exact same-type raw `ScalarValue`, including NaN payload/sign,
  infinity, signed zero, BOOL, and integer bits. Attribute type/raw bits participate in metadata,
  fingerprinting, the independent coordinate-map oracle, and public raw-value evidence.

Every route writes fresh canonical output storage and preserves all inputs. No raw movement
operation may use a typed MPSGraph selector as its production correctness argument.

### Gate D: explicit blockers

The following remain capability-false under both profiles:

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
5. Run public no-skip Metal-only Engine smokes for CAST, indexing, and layout, including raw NaN
   payloads, signed zeros, integer extremes, BOOL, rank-zero outputs, INT64 indices, invalid bounds,
   repeated runs, concurrent runs, lifecycle/close ordering, input preservation, output no-alias,
   and publication ownership.
6. Rebuild/sign/package the native library, verify the exact thirteen exports, run the focused Metal
   package and documentation/Javadoc checks, and do not run a final full repository build.
7. Commit implementation and focused evidence, request an independent cumulative Class C review,
   remediate every P0/P1/P2, rerun affected focused checks, and mark Complete only after approval.

Expected production capability is exactly `61 true / 54 false`: eleven newly true operation kinds
(`CAST`, three read-only indexing kinds, and seven copy-only layout kinds). If any named proof gate
fails, keep that kind false and record the exact reduced delta rather than weakening a gate.
