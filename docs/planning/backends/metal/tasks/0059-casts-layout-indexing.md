# Task 0059: Casts, [Layout](../../../../glossary.md#layout), and Indexing

## Reader context

A cast changes values under a conversion rule; indexing and raw movement copy selected bits. Native kernels need logical coordinates and physical geometry: SELECT/SLICE can describe holes or offsets even when publication returns a compact logical sequence. A Tensor description is not the buffer containing those bytes.

Java and native preflight independently check mapping and bounds before writes. Publication gathers logical values; transfer honors physical span and Boolean-byte validation. The six-carrier cast matrix, no-gradient boundary and schema-15 cutover below are historical, not today's full movement/cast domain.

- [Current logical versus physical authentication](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Explains offsets, spans and mutation boundaries.
- [Current cast indexing and movement admission](../../../../backend-guide/metal-backend.md#capability-whole-partition-analysis-and-route-selection) — Locates later carrier and gradient coverage.

## Status

Complete at [schema](../../../../glossary.md#term-schema)/[layout](../../../../glossary.md#layout) cutover plan `6478d249`, implementation and focused proof `783eabe1`,
package/schema documentation remediation `ac0db5c4`, [BOOL](../../../../glossary.md#data-type--datatype) [atomic](../../../../glossary.md#resource-transaction-and-rollback)-[publication](../../../../glossary.md#term-publication) proof `536e52b8`, and
cumulative review remediation `02a097eb`. Independent cumulative Class C review of
`6478d249..02a097eb` returned `APPROVE` with zero remaining P0/P1/P2 findings. The earlier
schema-14 implementation remains the audited baseline at `08bf68bb` with focused proof at
`a4926b52`.

## Change class

Class C. This was one serial [Metal](../../../../glossary.md#term-metal) capability, route, native decoder, exact custom-[kernel](../../../../glossary.md#kernel),
[preflight](../../../../glossary.md#term-preflight), mixed-program, public [Engine](../../../../glossary.md#term-engine), package, test, evidence, documentation, and narrow public
[CPU](../../../../glossary.md#term-cpu)/Metal storage-[layout](../../../../glossary.md#layout) transfer [API](../../../../glossary.md#term-application-programming-interface-api) cutover. It changes no [Model](../../../../glossary.md#term-model) semantics.

## Goal and exact inventory

Complete the structural [Metal](../../../../glossary.md#term-metal) recipes for the remaining cast, indexing, and [layout](../../../../glossary.md#layout) wires, and
admit production capability only for [occurrence](../../../../glossary.md#term-occurrence) domains implemented by an auditable exact custom
route:

- `39 CAST`;
- `69 GATHER_ELEMENTS`, `70 SCATTER_ADD`, `71 GATHER_ND`, `72 SCATTER_ND`, and `73 SELECT`; and
- `74 PAD`, `75 SLICE`, `76 SLICE_UPDATE`, `77 CONCAT`, `78 STACK`, `79 TILE`, `80 FOLD_AXIS`,
  `81 UNFOLD2D`, `82 FOLD2D`, `83 UNFOLD3D`, and `84 FOLD3D`.

Already approved `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, [`CONTIGUOUS`](../../../../glossary.md#term-contiguous),
[`GATHER`](../../../../glossary.md#gather), `ONE_HOT`, `SCATTER_ELEMENTS`, and `UNFOLD_AXIS` wires `6..19` are [regression](../../../../glossary.md#term-regression) [scope](../../../../glossary.md#term-scope)
only. This task must not replace, duplicate, broaden, or reinterpret their recipes or capability.

The starting ledger is exactly `50 true / 65 false`, `62 executable / 53 nonexecutable`, [schema](../../../../glossary.md#term-schema)
14, [identity](../../../../glossary.md#term-identity-meanings) 15, native [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, and thirteen exports.

## Frozen production gates

### Gate A: proved CAST pairs

`CAST` uses one deterministic integer/raw custom [kernel](../../../../glossary.md#kernel). Capability is [occurrence](../../../../glossary.md#term-occurrence)-sensitive and is
true only for the following explicitly proved [carrier](../../../../glossary.md#term-storage-carrier) pairs:

- all six same-type pairs, which are byte-identical copies;
- [`BOOL`](../../../../glossary.md#data-type--datatype) to each of [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), and [`INT64`](../../../../glossary.md#data-type--datatype), using the exact positive
  zero/one carrier encodings;<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- each non-BOOL source to `BOOL`, where false is exactly either floating zero sign or integer zero
  and every other raw carrier, including every NaN, is true;
- `INT32 -> INT64` exact [sign extension](../../../../glossary.md#term-sign-extension) and `INT64 -> INT32` exact low-bit [narrowing](../../../../glossary.md#term-narrowing); and
- `BFLOAT16 -> FLOAT32` exact sixteen-bit left [alignment](../../../../glossary.md#term-alignment).

Every other source/[target](../../../../glossary.md#term-target) pair remains capability-false. In particular, floating narrowing,
`FLOAT64` or `FLOAT32` [widening](../../../../glossary.md#term-widening)/conversion beyond the named pair, `BFLOAT16 -> FLOAT64`, and
integer-to-floating conversions are blocked until independent per-pair proofs cover [rounding](../../../../glossary.md#term-rounding),
[saturation](../../../../glossary.md#term-saturation), [signed zero](../../../../glossary.md#term-signed-zero), [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and [NaN](../../../../glossary.md#term-nan) sign/quiet/payload rules. No [MPSGraph](../../../../glossary.md#term-mpsgraph) cast
observation, sampled ordinary corpus, [tolerance](../../../../glossary.md#term-tolerance), or [host](../../../../glossary.md#term-host)-language cast authorizes them.

The admitted CAST domain permits [canonical](../../../../glossary.md#term-canonical-form) dense static [rank](../../../../glossary.md#term-rank) `0..16`, all positive extents for
positive rank, exact input/output [Shape](../../../../glossary.md#shape) equality, and `requiresGrad=false` on both sides. Every
potentially [gradient](../../../../glossary.md#term-gradient)-bearing cast occurrence remains false.

### Gate B: exact read-only indexing

Admit `GATHER_ELEMENTS` and `GATHER_ND` for exact raw payload movement:

- data/output may use any of the six [schema](../../../../glossary.md#term-schema) carriers;
- index [tensors](../../../../glossary.md#tensor) may use [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype) exactly as [Model](../../../../glossary.md#term-model) permits;
- [descriptors](../../../../glossary.md#term-descriptor) are [canonical](../../../../glossary.md#term-canonical-form) dense, fully static, zero-offset, non-[view](../../../../glossary.md#term-view-meanings), [rank](../../../../glossary.md#term-rank) `0..16` where the
  operation contract permits that rank, with positive extents for positive rank;<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- every input and output has `requiresGrad=false`;
- normalized axes, shared batch prefixes, tuple depths, Shapes, and selected-axis/result ranks must
  exactly match the Model-derived descriptors; and<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Shapes glossary](../../../../glossary.md#shape))</span>
- every index value is host-preflighted before command encoding, with no write or partial output on
  the first invalid bound.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>

The [kernel](../../../../glossary.md#kernel) maps one canonical output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) to one checked input coordinate and copies exactly
one [carrier](../../../../glossary.md#term-storage-carrier) payload. [Scalar](../../../../glossary.md#term-scalar) GATHER_ND results remain canonical rank zero.
Negative caller [axes](../../../../glossary.md#term-axis) are already normalized by Model/[Compiler](../../../../glossary.md#term-compiler); encoded negative or out-of-range
axes are malformed rather than renormalized in [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture).

### Gate C: exact copy-only [layout](../../../../glossary.md#layout)

Admit [`PAD`](../../../../glossary.md#padding), `CONCAT`, `STACK`, and `TILE` for all six [carrier](../../../../glossary.md#term-storage-carrier) representations. Admit
`UNFOLD2D` and `UNFOLD3D` for the [Model](../../../../glossary.md#term-model)-legal [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16)
representations. Inputs and outputs are [canonical](../../../../glossary.md#term-canonical-form) static [descriptors](../../../../glossary.md#term-descriptor) with exact Model
[Shapes](../../../../glossary.md#shape)/attributes and no [gradients](../../../../glossary.md#term-gradient).

- `PAD` copies original payloads and writes the exact same-type raw [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) for every padded
  coordinate.
- `CONCAT` and `STACK` preserve input order and use a bounded `1..16` input occurrence domain so
  every buffer is explicitly declared, bound, fingerprinted, and live. Scalar STACK is included;
  scalar CONCAT remains invalid by the Model contract.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- `TILE` implements whole-pattern per-axis repetition, not scalar run repetition. Model-legal
  scalar <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [run glossary](../../../../glossary.md#term-run))</span>`PAD` (empty widths) and [scalar](../../../../glossary.md#term-scalar) `TILE` (empty repeats) are exact one-[element](../../../../glossary.md#term-element) [identities](../../../../glossary.md#term-identity-meanings).
- `UNFOLD2D` and `UNFOLD3D` use exact [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) maps. Direct `Window2dAttrs`/`Window3dAttrs`
  write represented positive zero for out-of-domain samples; explicit <span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>`Unfold2dAttrs`/
  `Unfold3dAttrs` write the exact same-type raw `ScalarValue`, including NaN payload/sign,
  infinity, and signed zero. Model excludes BOOL and integer carriers from both window kinds.
  Attribute type/raw bits participate in metadata, fingerprinting, the independent coordinate-map
  oracle, and public raw-value evidence.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [signed zero glossary](../../../../glossary.md#term-signed-zero); [BOOL glossary](../../../../glossary.md#data-type--datatype); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [metadata glossary](../../../../glossary.md#term-metadata); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

Every route writes fresh canonical output storage and preserves all inputs. No raw movement
[operation](../../../../glossary.md#operation) may use a typed [MPSGraph](../../../../glossary.md#term-mpsgraph) selector as its production correctness [argument](../../../../glossary.md#term-argument).

### Gate D: [schema](../../../../glossary.md#term-schema)-15 resolved-[layout](../../../../glossary.md#layout) indexing

Admit `SELECT` and [`SLICE`](../../../../glossary.md#slice) through the existing exact raw-movement custom family for all six
carriers, with fully static positive-[rank](../../../../glossary.md#term-rank) input and output [Shapes](../../../../glossary.md#shape), positive extents,
`requiresGrad=false`, and resolved positive-[stride](../../../../glossary.md#term-stride) non-overlapping storage layouts throughout.
Caller-negative [axes](../../../../glossary.md#term-axis) and SELECT indices must already be normalized by [Model](../../../../glossary.md#term-model); the executable image
accepts only normalized attributes. SELECT removes one axis. SLICE accepts normalized starts,
lengths, axes, and strictly positive steps. Zero or negative steps, unresolved layouts, zero
strides, overlapping geometry, and a [referenced span](../../../../glossary.md#referenced-element-span) inconsistent with the offset/strides [fail
closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation); neither capability nor contextual [partition](../../../../glossary.md#partition) [admission](../../../../glossary.md#term-admission) promises those cases.

Schema 15 encodes every logical layout fact needed to validate, execute, transfer, and publish
these materialized results: Shape rank and [dimensions](../../../../glossary.md#dimension), optional-layout presence, every resolved
[element stride](../../../../glossary.md#element-stride), [element](../../../../glossary.md#term-element) [storage offset](../../../../glossary.md#term-storage-offset), derived layout kind, [view](../../../../glossary.md#term-view-meanings) flag, and [referenced element
span](../../../../glossary.md#referenced-element-span). [Native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture) independently recomputes and validates kind/span geometry and the exact
SELECT/SLICE layout transformation. Result [buffers](../../../../glossary.md#term-buffer) cover the complete physical referenced span.
The custom [kernels](../../../../glossary.md#kernel) read and write only logical [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) at their declared physical positions,
so prefix canaries and layout holes remain unchanged. Internal values and cross-[owner](../../../../glossary.md#term-owner) boundaries
retain the same exact storage-layout contract; [publication](../../../../glossary.md#term-publication) gathers logical elements in [row-major](../../../../glossary.md#term-row-major)
order without rewriting the [descriptor](../../../../glossary.md#term-descriptor) or assuming dense storage. This applies to all six [carrier](../../../../glossary.md#term-storage-carrier)
types and does not [alias](../../../../glossary.md#aliasing-and-arena) or mutate the input.

The clean cutover changes schema 14 to schema 15 and [identity](../../../../glossary.md#term-identity-meanings) 15 to identity 16. There is one
schema-15 encoder/decoder and no schema-14 compatibility reader, alias, migration shim, or dual
path. Native [ABI](../../../../glossary.md#term-application-binary-interface-abi) remains 5 and the export [signature](../../../../glossary.md#term-signature)/count remains unchanged.

### Gate E: explicit blockers

The following remain capability-false under both profiles:
- `SCATTER_ADD` and `SCATTER_ND`, because duplicate membership/order-independent reductions,
  replacement uniqueness, per-type reduction semantics, and gradient-bearing domains are not all
  proved by a single systematic route;<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- `SLICE_UPDATE`, because functional replacement plus base/update gradient partition and overlap
  obligations are not proved by the read-only copy kernel;<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- `FOLD_AXIS`, `FOLD2D`, and `FOLD3D`, because overlap accumulation is arithmetic and inherits the
  duplicate/reduction result-set and gradient blockers; and
- every otherwise exact new operation occurrence with gradient-bearing metadata, a dynamic or empty
  descriptor, rank-zero SELECT/SLICE result, an unproved CAST pair, unsupported variadic
  cardinality, or malformed attributes/layout geometry.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [occurrence glossary](../../../../glossary.md#term-occurrence); [metadata glossary](../../../../glossary.md#term-metadata); [descriptor glossary](../../../../glossary.md#term-descriptor); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [SLICE glossary](../../../../glossary.md#slice); [variadic glossary](../../../../glossary.md#term-variadic); [cardinality glossary](../../../../glossary.md#term-cardinality); [layout glossary](../../../../glossary.md#layout))</span>

Do not special-case a nonoverlap geometry to make a blocked reduction kind true. Do not use
atomics, encounter order, [tolerance](../../../../glossary.md#term-tolerance), selector success, or ordinary-value output as proof.

## Structural [MPSGraph](../../../../glossary.md#term-mpsgraph) recipes

Implement native-validated structural recipes for all seventeen wires independently of production
capability:

- direct cast for `CAST`;
- [gather](../../../../glossary.md#gather)-along-[axis](../../../../glossary.md#term-axis) for `GATHER_ELEMENTS`, direct [gather-ND](../../../../glossary.md#gather-nd) for `GATHER_ND`, and slice-plus-squeeze
  for `SELECT`;
- direct scatter/index selectors where the installed [API](../../../../glossary.md#term-application-programming-interface-api) expresses `SCATTER_ADD` and `SCATTER_ND`;
- direct [constant](../../../../glossary.md#term-constant) pad, strided [slice](../../../../glossary.md#slice), [slice update](../../../../glossary.md#slice-update), concat, and tile selectors;
- expand-[dimensions](../../../../glossary.md#dimension) plus concat for `STACK`;
- the catalog scatter-add composition for `FOLD_AXIS`;
- direct [im2col](../../../../glossary.md#term-im2col)/[col2im](../../../../glossary.md#term-col2im) for `UNFOLD2D`/`FOLD2D`; and
- the catalog's explicit 3D window compositions for `UNFOLD3D`/`FOLD3D`.

When an installed MPSGraph selector has a narrower [rank](../../../../glossary.md#term-rank)/[cardinality](../../../../glossary.md#term-cardinality) contract than [Model](../../../../glossary.md#term-model), the
structural [fixture](../../../../glossary.md#term-fixture) must validate and document that [occurrence](../../../../glossary.md#term-occurrence) boundary rather than silently claim
production support. Structural selector presence, executable creation, and bounded output are never
production semantic proof.

After implementation, the structural ledger must be exactly `79 executable / 36 nonexecutable`.
The catalog counts remain `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; only their recipes become
native-executable.

## Custom route and mixed-program design

Add one systematic data-movement custom family plus the bounded proved CAST [matrix](../../../../glossary.md#term-matrix-meanings). The fixed
preparation decision is `CUSTOM_PROGRAM` for every newly admitted [occurrence](../../../../glossary.md#term-occurrence). It is made once,
before resource declaration, and is bound into candidate [identity](../../../../glossary.md#term-identity-meanings) and workload [fingerprint](../../../../glossary.md#term-fingerprint).

The custom [metadata](../../../../glossary.md#term-metadata) binds [operation](../../../../glossary.md#operation) wire, source/[target](../../../../glossary.md#term-target) [carrier](../../../../glossary.md#term-storage-carrier) widths and types, every [Shape](../../../../glossary.md#shape),
[normalized axis](../../../../glossary.md#normalized-axis)/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/tuple fact, complete attributes, exact [scalar](../../../../glossary.md#term-scalar) bits, [variadic](../../../../glossary.md#term-variadic) input order and
count, and checked [dispatch](../../../../glossary.md#term-dispatch) geometry. Variable inputs are explicit [liveness](../../../../glossary.md#term-liveness)/resource references;
there is no hidden [buffer](../../../../glossary.md#term-buffer) [lookup](../../../../glossary.md#term-lookup). Mixed programs may combine these [nodes](../../../../glossary.md#node) with every already approved
custom or nested [MPSGraph](../../../../glossary.md#term-mpsgraph) node. Every intermediate is declared once, allocated once, and consumed
through the existing isolated workspace and direct assigned-output [publication](../../../../glossary.md#term-publication).

There is exactly one preparation-fixed route: no fallback, retry, alternate selector after failure,
[runtime](../../../../glossary.md#term-runtime) probing, timing, [benchmarking](../../../../glossary.md#term-benchmarking), [autotuning](../../../../glossary.md#autotuning--model-autotuning), or [device](../../../../glossary.md#term-device)-dependent choice.

## [Schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), [ABI](../../../../glossary.md#term-application-binary-interface-abi), [lifecycle](../../../../glossary.md#lifecycle), and errors

The reopened inventory found that schema 14 encoded only type, [Shape](../../../../glossary.md#shape) [rank](../../../../glossary.md#term-rank), [dimension](../../../../glossary.md#dimension)-pool offset,
and [gradient](../../../../glossary.md#term-gradient) flag for each value; it carried no [layout](../../../../glossary.md#layout)-presence, [stride](../../../../glossary.md#term-stride), storage-offset, kind, [view](../../../../glossary.md#term-view-meanings),
or referenced-span fact. The [publication](../../../../glossary.md#term-publication) path authenticated only [FLOAT32](../../../../glossary.md#term-float32), zero-offset affine
outputs before downloading the dense represented-order [buffer](../../../../glossary.md#term-buffer). That representation cannot publish
the exact [Model](../../../../glossary.md#term-model) [descriptors](../../../../glossary.md#term-descriptor) produced by SELECT and positive-step [SLICE](../../../../glossary.md#slice).

Perform one clean schema-15/identity-16 cutover. The schema-15 value descriptor and shared stride
pool encode the complete optional logical layout listed in Gate D while retaining [operation](../../../../glossary.md#operation) wires
`1..115`, type wires `1..6`, attribute wires `0..41`, native ABI 5, and exactly thirteen exports.
Every image offset/count calculation is checked and schema 14 is rejected exactly; there is no
compatibility reader.

Native parsing independently validates every later attribute kind used here, [canonical](../../../../glossary.md#term-canonical-form) unused
fields, counts, rank bounds, positive dimensions, signed attribute words, [carrier](../../../../glossary.md#term-storage-carrier) compatibility,
complete Shape equations, [topology](../../../../glossary.md#term-topology), and route legality. Malformed schema and raw-ABI tests cover
truncation, wrong kinds/counts/types, invalid [axes](../../../../glossary.md#term-axis)/[batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/tuple depths/steps/windows/[scalars](../../../../glossary.md#term-scalar), excess
[variadic](../../../../glossary.md#term-variadic) inputs, and illegal custom/[MPSGraph](../../../../glossary.md#term-mpsgraph) route mixing.

Index bounds failures must map to the existing stable range failure without command submission or
partial output. All other invalid [occurrence](../../../../glossary.md#term-occurrence) [metadata](../../../../glossary.md#term-metadata) fails before allocation or native creation.
Preserve transactional finalization, reverse/attempt-all [rollback](../../../../glossary.md#term-rollback), child [leases](../../../../glossary.md#term-lease), [synchronous](../../../../glossary.md#term-synchronous) runs,
repeated and concurrent execution, close-vs-[run](../../../../glossary.md#term-run) ordering, no-[alias](../../../../glossary.md#aliasing-and-arena) input/output resources, input
immutability, and direct assigned-output publication.

## Verification and acceptance

1. Freeze this plan before production edits.
2. Add an independent CAST-pair proof/checker that does not share the production transform. It must
   exhaust both BOOL values and all 65,536 BFLOAT16 values for the admitted BFLOAT16 widening and
   BOOL target behavior, partition INT32/INT64 sign-extension/narrowing/zero behavior with exact
   boundary and randomized full-word corroboration, and prove identity pairs by carrier-width copy.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [widening glossary](../../../../glossary.md#term-widening); [target glossary](../../../../glossary.md#term-target); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [narrowing glossary](../../../../glossary.md#term-narrowing); [identity glossary](../../../../glossary.md#term-identity-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
3. Prove raw coordinate maps against an independent CPU/Model oracle across every admitted kind,
   every carrier width, scalar/rank boundaries, padding, batch/suffix gather, window
   padding/dilation/stride, variadic placement, and positive slice steps; separately prove that
   zero/negative steps and unresolved, zero-stride, overlapping, or out-of-span layouts fail closed.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [oracle glossary](../../../../glossary.md#term-test-oracle); [scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [padding glossary](../../../../glossary.md#padding); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [gather glossary](../../../../glossary.md#gather); [dilation glossary](../../../../glossary.md#term-dilation); [stride glossary](../../../../glossary.md#term-stride); [variadic glossary](../../../../glossary.md#term-variadic); [layouts glossary](../../../../glossary.md#layout); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
4. Run focused capability/catalog/schema/identity/malformed-native tests and assert the exact final
   ledger. No previously approved operation may change route or domain.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [schema glossary](../../../../glossary.md#term-schema))</span>
5. Run public no-skip Metal-only Engine smokes for every admitted CAST, indexing, and layout kind,
   including signed zeros, integer extremes, normalized negative caller axes, repeated runs,
   concurrent runs, lifecycle/close ordering, input preservation, output no-alias, and publication
   ownership. Raw native fixtures separately cover all 65,536 BFLOAT16 words, deterministic
   integer full-word corpora, INT64 indices, and bounds-before-write failures.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [axes glossary](../../../../glossary.md#term-axis); [lifecycle glossary](../../../../glossary.md#lifecycle); [alias glossary](../../../../glossary.md#aliasing-and-arena); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
6. Rebuild/sign/package the native library, verify the exact thirteen exports, run the focused Metal
   package and documentation/Javadoc checks, and do not run a final full repository build.<span data-glossary-navigation="true"> ([native library glossary](../../../../glossary.md#term-native-library); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
7. Commit implementation and focused evidence, request an independent cumulative Class C review,
   remediate every P0/P1/P2, rerun affected focused checks, and mark Complete only after approval.<span data-glossary-navigation="true"> ([Commit glossary](../../../../glossary.md#term-commit))</span>

Expected production capability after the reopened cutover is exactly `61 true / 54 false`: the
previous nine newly true [operation](../../../../glossary.md#operation) kinds plus `SELECT` and [`SLICE`](../../../../glossary.md#slice). The structural ledger remains
exactly `79 executable / 36 nonexecutable`; their structural [MPSGraph](../../../../glossary.md#term-mpsgraph) recipes are independent of
the fixed custom production route.

## Completion evidence

- Production capability is exactly `61 true / 54 false`: the prior nineteen CAST pairs, exact
  `GATHER_ELEMENTS`/`GATHER_ND`, six copy-only layout routes, plus SELECT and positive-step SLICE.
  The reduction/update/fold, gradient, unresolved, zero/negative-stride, overlapping, out-of-span,
  empty/dynamic, and unproved conversion domains remain false.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [SLICE glossary](../../../../glossary.md#slice); [gradient glossary](../../../../glossary.md#term-gradient); [stride glossary](../../../../glossary.md#term-stride))</span>
- Structural execution remains exactly `79 executable / 36 nonexecutable`; catalog state remains
  `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE` and custom state remains
  `38 AVAILABLE / 77 PENDING / 0 UNAVAILABLE_WITH_PROOF`.
- Schema 15 encodes 40-byte value descriptors plus the stride pool, offset, kind, view flag, and
  referenced span. Java and native reject schema 14, malformed flags/kinds/spans, unresolved
  layouts, zero/negative strides, and overlap. All backend-local identity versions are 16; version
  15 fails closed. ABI 5 and exactly thirteen exports remain unchanged.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [descriptors glossary](../../../../glossary.md#term-descriptor); [view glossary](../../../../glossary.md#term-view-meanings); [referenced span glossary](../../../../glossary.md#referenced-element-span); [backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Real native SELECT-to-SLICE chaining passed for every carrier with nonzero offsets,
  noncontiguous positive strides, exact represented bits, repeated execution, unchanged inputs,
  and untouched prefix/gap canaries. Focused native schema, malformed-image, indexing,
  rank-sixteen, and existing structural controls also passed.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [represented bits glossary](../../../../glossary.md#term-represented-bits); [rank glossary](../../../../glossary.md#term-rank))</span>
- Public storage-layout transfer covers SELECT and SLICE boundaries, preserves complete physical
  spans, and rejects unresolved, zero-stride, negative-stride, overlapping, and byte-overflowing
  geometry. Metal-to-CPU BOOL download uses cold private staging and validates every logical byte
  before committing, so a malformed device value leaves the caller destination unchanged.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [CPU glossary](../../../../glossary.md#term-cpu); [BOOL glossary](../../../../glossary.md#data-type--datatype); [device glossary](../../../../glossary.md#term-device))</span>
- The public Engine partition boundary admits a graph-input storage layout through forced
  SELECT-to-SLICE owner boundaries and rejects unsupported transfer geometry. Authenticated
  SELECT publication gathers exact canonical bits for all six carriers.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [partition glossary](../../../../glossary.md#partition); [graph glossary](../../../../glossary.md#graph); [owner glossary](../../../../glossary.md#term-owner); [publication glossary](../../../../glossary.md#term-publication); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- The dylib was rebuilt, fixed-signature ad-hoc signed, locally packaged, and independently
  verified with ABI 5, schema 15, and thirteen exports. The complete Metal JVM suite, focused
  real-native suites, CPU transfer API proof, Engine boundary proof, and changed-module Javadocs
  passed. Per plan, no final full repository build was run.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [signature glossary](../../../../glossary.md#term-signature); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run))</span>
- Independent cumulative Class C review covered `6478d249..02a097eb` and returned `APPROVE` with
  zero remaining P0/P1/P2. Remediation closed package-manifest/schema drift, stale current
  documentation, transfer-predicate byte overflow, misleading raw-borrow documentation, hot BOOL
  staging allocation, and strengthened the atomic no-partial-publication proof.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [overflow glossary](../../../../glossary.md#term-overflow); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
