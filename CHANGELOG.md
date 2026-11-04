# Changelog

## 0.1.0 — private/internal

### Added

- Explicit oneDNN 3.14.0 LP64 sequential CPU composition with `openOneDnn(Path)` and immutable
  `AUTO`/`PRIMITIVES`/`GRAPH` API preferences inside the existing CPU owner. Required explicit
  load failures throw; default CPU/OpenBLAS discovery is unchanged and no binary is bundled.
  The bare homogeneous static dense/permuted-dense FLOAT32 MATMUL baseline, including normalized
  broadcast batches up to rank 12 with K > 1, now also admits qualified F32 common arithmetic,
  predicates, aggregates, composed raw movement and literal MSE/BatchNorm/padded-average-pool
  units. Unsupported API/implementation/geometry retains its preplanned CPU unit; opaque
  convolution/normalization/activation replacements and Scan orchestration remain unqualified.
  Literal SUM/DIV preserves mean divisors without reciprocal substitution; nonexact F32 counts
  reject cold. Singleton-stride normalization prevents primitive/Graph native-cache aliasing.
  Shape-identity reductions reject before native queries to avoid the pinned Graph backend's
  unsupported reduction-descriptor abort.
  Common-IR whole-plan heuristic comparison,
  declared ingress/scratch, per-run current-weight refresh and leased native resources preserve
  saved/public stores and existing AD; Phase-2 schema 4 rejects stale policy/session decisions.
  Actual macOS arm64 and Linux x86-64 native/public qualification and report-only training
  baselines are recorded in the [configuration and exact binary pins](docs/backend-guide/cpu-backend.md#explicit-onednn-composition).

- Qualified shared Metal FLOAT32/BFLOAT16/FLOAT16 elementary, scalar/tensor power,
  masked/statistical reduction and cumulative execution, including rank-zero values, affine
  scalar/clamp/reduction reads and generated expanded-one counts. Numerical bodies use FLOAT32
  working sites and original-carrier publication; cumulative INT32/INT64 arithmetic is modular.
- Metal normalization, categorical losses and attention now share original-carrier FLOAT32,
  BFLOAT16 and FLOAT16 kernel bodies, with explicit per-role metadata, Model promotion and
  typed epsilon/scale bits. Attention keeps scores/probabilities in declared FLOAT32 work
  storage until each final output is narrowed; low weights siblings do not replace that work.
  Independent raw validation preserves canonical batch-statistic siblings, labels, masks and
  exact gradient role flags. FLOAT16 attention accepts its documented FLOAT32 scale metadata
  in both public construction and captured inference; CPU low arithmetic restrictions are unchanged.
- Removed the Metal CONCAT/STACK sixteen-operand binding limit through dynamic references and
  disjoint copies directly into final storage, shared by ordinary and controlled Scan execution.
  Actual public GPU qualification covers 17/33/257 whole inputs, last-operand-only gradients,
  empty outputs/cotangents and 33 distinct Scan operands in both directions and activity modes
  across FLOAT32/BFLOAT16/FLOAT16.
- Native Metal general Scan through authenticated bounded body/binder plans and once-prepared
  ordinary kernels, including actual reverse bodies and runtime-length RNN training/save/load/
  fresh training. Structural inactivity guards operands before loads or index validation;
  there is no Java body loop, time unrolling or CPU fallback.
- Original-typed Metal exact stored F64/integral operations and promotion, independent Model
  metadata inference, canonical multi-output sibling retention and source-only gradient
  publications through the existing initializer and per-run resource owner.
- First-order fixed RNN_TANH, GRU_RESET_AFTER and LSTM gradients through the shared length-aware
  Scan/ordinary formula owners, including sequence/final-state seeds, both directions and empty
  time/batch domains. Fixed Metal forward uses the common F32/BF16/F16 numerical codec with
  F32 low-carrier working sites and explicit recurrent-state narrowing. CPU low arithmetic
  admission and fixed recurrent higher-order restrictions remain unchanged.

- Model-valid CPU promoted F32/F64 and I32/I64 pointwise operands, exact comparisons and selected
  WHERE conversion, retaining each gradient target's original carrier.
- Checked CPU signed/empty slice materialization and its existing extraction/update pullbacks,
  bounded one-pass variadic CONCAT/STACK beyond sixteen inputs, and shared sound output
  injectivity proofs for large interleaved layouts.

- Shared Metal FLOAT32/BFLOAT16/FLOAT16 matmul, convolution, pooling and overlapping
  axis/image fold execution owners, with FLOAT32 working sites and one final low-format
  conversion after actual additions; single-contributor fold sites preserve original raw words.
  General scatter reductions and Gather cotangents retain original value carriers,
  multidimensional indices and gradient flags. Raw signed slices materialize all seven carriers;
  mixed floating comparisons use exact represented-value ordering. Metal FLOAT64 folding remains
  non-overlapping and CPU BFLOAT16/FLOAT16 arithmetic restrictions remain unchanged.
- Fluent `ModelBuilder.maxPool3d(kd,kh,kw)` and `averagePool3d(kd,kh,kw)`, with kernel stride,
  zero padding, unit dilation and floor geometry; ordinary Conv3d/pooling/linear Model training
  and save/load continue through existing owners with freshly configured training.
- Complete typed Pool3d cold geometry, including otherwise-qualified all-valid and static
  homogeneous FLOAT32/FLOAT64 length-aware Scan bodies. Only these two constructors are added
  to the narrow length whitelist; existing role, axis and terminal restrictions remain.
- Append-only `graph-model.v6` (`SYNGRF06`, version 6), opcodes 34/35 for maximum/average Pool3d,
  twelve i64 geometry fields and a canonical ceil flag. Contained Pool3d also selects v6;
  unchanged sources keep byte-identical smallest v1–v5 encoding and no optimizer/session state.
- Metadata-only `GraphCompilationPort.validateSource` shares complete cold preflight without
  Tensor identities or host copying. Decode now rejects invalid metadata before creating any
  candidate Tensor for every supported version.
- The Pool3d checkpoint advanced Metal identity to40 and the previous audit used identity41/
  schema21. The shared numerical cutover uses identity42/schema22/header124, retaining operation116/
  attribute42 for Scan and ABI7/thirteen exports.
- Static Model-legal per-role FLOAT32/FLOAT64/BFLOAT16/FLOAT16 general Scan cold capture,
  including legal casts, Checkpoint copy/rebind and NN acquisition/cold-forward paths.
  Reconstruction preserves explicit row roles, aliases and canonical ports without unrolling
  time, reading external lengths or loosening exact rebind descriptors. Symbolic cold bodies
  remain outside this domain; Metal FLOAT64 numerical execution is not implied.
- Existing selective graph-model.v5/v6 persistence retains row roles, aliases, canonical ports
  and named values without storing external length batches or changing checkpoint wire versions.
  Loaded typed Models accept fresh lengths and newly configured training.
- Explicit `RnnSequence.forwardGeneralScan` and trainable `rnnScan` Models support all four
  floating carriers through the ordinary RnnCell equation and initialization. Existing low
  default builders retain fixed-cell working/publication semantics. Semantic final-hidden
  selection supports general/fixed ports; fresh-length training and save/close/load/new training
  use the existing execution owners.
- True empty-carry Scan source and explicit lengthAware metadata, including zero-ROW Shared
  source with B derived from lengths. All canonical typed ports/aliases remain; source validity,
  selected derivatives, backend execution and cold/wire admission remain separate checks.
- Descriptor-only ordinary checking through ScanBody.describeOrdinary, sharing existing Tensor
  family guards and all canonical siblings without allocating Tensor identities. Source checking
  does not widen length-aware row proof, AD, backend or persistence support.
- Versioned local Maven, binary and exact-source distributions for eighteen implemented products,
  with source/Javadoc JARs, real public dependency closure, third-party notices, explicit source
  provenance and reproducible source archives. External Maven/Gradle CPU/Metal model consumers
  run without checkout JARs. The bounded private/internal 0.1.0 matrix is technically qualified
  through the root gate, isolated Maven replay and clean eighteen-product NoGit/source-native
  rebuilds. Public project licensing, registry upload and trusted native-prebuilt signing are separate.
- Shared per-output ordinary Scan coordinate metadata, including canonical BatchNorm/attention
  dependencies, completely proved nonleading intermediate rows and terminal contributor routing.
  Model/raw forward validation reuse one seam; retained AD/header/physical/cold/wire boundaries
  remain independent and unrepresentable saved histories reject before derivative identities.
- General CPU Scan composition through existing ordinary lowerers and emitters, for all-valid and
  length-aware static/qualified one-B bodies with per-value types, exact selected first-order VJPs,
  structural absence distinct from present zero and run-private physical/control storage.
  Genuine qualified legacy keys, ABIs and generated bytes remain replayable. This adds no second
  executor, broader cold/wire format, Metal Scan capability or Java low-arithmetic unlock.
- Trainable grouped Conv3d through shared Conv2d/Conv3d unfold/MATMUL/overlap-add-fold adjoints:
  CPU FLOAT32/FLOAT64 and Metal FLOAT32 have every-input/weight/bias finite-difference proof,
  including biased grouped anisotropic and unbiased depthwise cases, plus real NN training,
  bit-exact parameter save/close/load and newly configured training. CPU general Scan composes
  the complete Conv3d forward/reverse body in both types, modes and traversal directions.
  At that checkpoint CPU Pool3d required no-gradient occurrences; the Pool3d addition above
  supersedes that gate. Ordinary public CPU named-B Conv3d, CPU low arithmetic and Metal Scan
  are not widened by either backend change.

### Changed

- Live CPU/Metal audits cover all 115 ordinary operation kinds, 137 signatures and separate Scan,
  classifying execution, selection, illegal projections and nondifferentiable/deferred roles.
  Each Metal F32/BF16/F16 family executes the same 187 forward and 358 selected-gradient cases
  without a Model-legal selection gap. Execution counts do not substitute for independent
  numerical and public model/training/checkpoint checks.
- One Metal fixed-corpus authentication covers 23 components/187,750 UTF-8 bytes and the common
  original-carrier codec; the separate low library and frozen FLOAT32 descriptor proxy are retired.
  General FLOAT64 arithmetic and CPU low arithmetic are not added.
- Temporarily lock Java CPU BFLOAT16/FLOAT16 numerical computation at source, preparation,
  represented-IR, direct generation/definition, retained cache and fixed-recurrent admission.
  Keep the implementation, exact/raw/discrete/storage/conversion roles and explicit F32 compute/
  F32-master training. Metal low support, shared types and old qualified generated identities remain.
  No runtime unlock, hidden low emulation or blanket low-containing-partition rejection is added.
- Ordinary CPU generation shares one namespaced method-emission dispatch for standalone classes
  and same-class helpers, including existing loss and row-phase methods. Standalone generated
  identities and class bytes remain unchanged; this factoring does not widen Scan execution.
- All eleven ordinary derivative families share one formula body for direct Tensor construction
  and identity-free checked metadata construction. Model guards and independent Compiler inference
  remain distinct; exact constants, lexical primal bindings and optional eager-site observation
  are preserved. This does not itself widen Scan admission or add a derivative implementation.
- Compiler and CPU local projection share ordinary computed logical-layout completion on Model
  ScanBody. Existing affine/crop geometry, materializing-RESHAPE decisions and the separate generic
  symbolic classifier have one owner; captured graphs, ordinary inference and strict ingress
  guards remain unchanged.
- CPU generated raw F32/F64/BF16/F16 UNFOLD3D and arithmetic F32/F64 FOLD3D share existing
  movement/fold owners, preserving qualified legacy identities and the low arithmetic lock.

### Fixed

- CPU FLOAT32 RSQRT preserves its square-root working site before reciprocal across scalar,
  vector/remainder and composed helpers; selective schema 76 rejects stale affected bodies.
- CPU dropout now uses typed complement/reciprocal/multiply with numeric identity v2, preserving
  exact RNG words, masks and next-state. SOFTMAX/LOG_SOFTMAX algorithm 3 handles the Model's
  nonfinite and empty-slice domain without a separate finite-only validator.
- CPU attention preserves zero-plus-singleton batch broadcasting. Wide CONCAT/STACK uses a
  prepared operand table and checked geometry under selective schema 75 instead of an N-wide
  JVM entry budget; small existing entries and controlled arena ownership remain unchanged.
- Metal semantic identity 44 aligns affine unary/comparison/MATMUL capability, preparation and
  raw preflight, closes ordinary empty and zero-grid fused-anchor execution, and maps proved
  numeric Scan ingress through existing CONTIGUOUS. Canonical non-gradient raw F64 Scan stays
  source-only; ABI7/schema22 and intentional numerical/resource restrictions remain.
- Cold carry-free and zero-ROW/scalar-SHARED Scan reconstruction/persistence uses the existing
  v4/v5/v6 records, deriving batch geometry from the external lengths descriptor.
- Owned fixed and mixed recurrent Models now train through the existing TrainingSession and
  Compiler first-order AD, retaining original recipes/carriers and fresh-training-after-load
  semantics. CPU low arithmetic and higher-order recurrence are not unlocked.
- CPU qualification no longer requires macOS temporary paths, a particular vector width,
  generated-code hashes or latency thresholds. Numerical controls retain exact non-NaN
  comparisons, typed overflow, subrange/canary checks and the selected StrictMath TANH
  producing site; arithmetic NaNs are checked by class rather than unspecified payload bits.
- CPU one-term MATMUL preserves its exact FLOAT32/FLOAT64 product in scalar, vector, tiled,
  ordinary-batch and controlled/regional Scan execution. Empty/inactive sums retain positive
  zero and multi-term typed sites remain unchanged. Selective artifact identity rejects stale
  contraction bodies; unqualified OpenBLAS singleton contractions remain on the planned CPU path.
  Numerical regressions replace obsolete literal ClassFile/opcode snapshots.
- Metal one-term MATMUL preserves the typed product's signed zero without an added zero or
  FMA, across FLOAT32/BFLOAT16/FLOAT16. The shared custom owner guards both operands before
  loads; the existing rank-two graph owner uses exact singleton broadcast multiplication.
  Semantic identity advances to43; ABI7, program schema22 and multi-term arithmetic stay unchanged.
- Metal signed SLICE materializes all seven stored carriers without negative physical strides;
  numerical extraction/update pullbacks use existing owners. Overlapping floating FoldAxis works
  in controlled Scan rows and guards inactive NaN contributors.
- Genuine empty storage/EXPAND detects zero extents before product/stride checks, including
  zero-grid native execution without a load or dispatch. Scalar values remain one stored element.
- Scalar native graph arithmetic restores declared rank after literal broadcasting; original
  FLOAT32 FLOOR/CEIL/SIGN admission agrees with native preflight and existing exact kernels.
- FLOAT16 normalization inference accepts the Model's FLOAT32 epsilon/momentum metadata;
  its existing normalization adjoints use FLOAT32 working expressions and publish original
  target-carrier gradients. BFLOAT16/FLOAT32/FLOAT64 metadata matching remains exact.
- Dropout capability checks original F32/BF16/F16 value/output descriptors while preserving
  canonical BOOL mask and INT64 state roles.
- Host snapshots resolve canonical geometry for fully static empty publications with no logical
  layout hint, preserving selected-backend ownership, leases and byte budgets. Nonempty unresolved
  publications remain rejected.
- Metal ordinary transport checks words/spans and ordering/reduction/fold axes before inference.
  Malformed FoldAxis target axes reject deliberately before indexing the declared shape.
- FLOAT32 arithmetic over affine operands now uses existing physical-stride kernels rather than
  packed MPSGraph feed binding. Offset extraction cotangents retain every contribution when a
  target is repeated in STACK, without new math or an extra copy; canonical routes are unchanged.
- CPU unit contraction now proves convexity on the current quotient graph, preventing cyclic
  prepared schedules after earlier contractions. Oversized variadic candidate inventories use
  existing direct decomposition without relaxing fusion or generated-method resource limits.
- Empty affine generated ranges guard before address/carrier reads; selective projection 73
  invalidates affected old dense-array classes without advancing generator envelope 71.
- Target-relative crop inference retains raw Model descriptors until physical layout completion,
  so actual extraction adjoints accept their original output contract.
- CPU attention follows literal eligible-score EXP normalization rather than shifted softmax.
  Its nonempty weighted-value contractions and fixed-recurrent projections seed their first
  product, preserving negative zero; empty/exclusion guards remain. Attention projection 74 rejects
  obsolete shifted classes and binds the new working-site marker in both identity streams.
- Current generated-code structural oracles consume live provider/preparer/generator facts instead
  of golden class-hash and row-count snapshots. Typed ABI, direct-loop and paired-Java checks remain,
  including avoidable-dispatch guards; historical checkpoint data is unchanged.

- General Scan preserves rank-17 temporal containers around rank-16 ordinary step values.
  The ordinary rank limit applies to every local value; exact outer role binding retains the
  single time axis without widening ordinary numerical ranks.
- Default scalar CPU preparation and MATMUL discovery no longer link an unrequested incubator
  Vector API module. Available vector alternatives remain intact when the module is resolved;
  explicit vector requests keep their existing policy. Scalar kernels and identities are unchanged.
- CPU FLOAT64 SOFTMAX/LOG_SOFTMAX now accepts all finite represented inputs, including internal
  negative-infinity maximum shifts, matching FLOAT32 and the existing stable Model formula.
  Nonfinite inputs still reject before consumer writes; arithmetic, saved-output derivatives,
  loss/attention formulas and generated/cache identities are unchanged.
- Composite Scan emits the correct interface-static MemorySegment.copy reference and advances
  only composite cache identities so trusted pre-fix envelopes cannot restore the invalid call.
  Qualified non-composite keys, entry ABIs and generated bytes remain unchanged.
- Ordinary aggregate, advanced/masked reduction and loss geometry compare retained primitive
  arrays by value. Repeated preparation and live artifact reuse retain exact guards instead of
  rejecting equivalent independently lowered physical facts.
- Complete graph preparation declares producerless published caller values through the existing
  source-resource path, including direct caller-seed gradients at T=0. Caller slots retain Compiler
  order and borrowed lifetime; constants retain existing initialization. Source-only named B,
  including zero-byte templates, coexists with genuinely fixed CPU work without changing its kernel
  geometry or relaxing Runtime defaults. Raw seven-type publication remains separate from arithmetic.
- Advanced Engine run publication now rolls back wrapper/registration failures and always releases
  admission, including cleanup errors. Closed Engine/Runtime results and states release direct
  fresh-payload references without closing caller storage; metadata, counts/geometry and existing
  failure-replay boundaries remain unchanged.
- Ordinary AD construction consistently resolves lexical value operands and saved canonical
  statistics/indices through its existing binding protocol. Reduction shares one complete ordinary
  implementation; formulas, contribution identities, eligibility and current Scan bounds remain.
- Final Compiler layout closure now distinguishes a proved RESHAPE view from an ordinary static
  materializing result after noncontiguous input. Exact views and raw source/wire inference remain
  unchanged; no derivative rewrite or consumer CONTIGUOUS is inserted. Backend admission remains separate.
- Java CPU ordinary static noncontiguous-input RESHAPE now uses the existing affine ordinal copy
  for canonical non-view materialization, including exact low/INT/BOOL movement. Genuine views,
  checked addressing, old qualified generated identities and the low-arithmetic lock are unchanged.
- Metal dense logical views now compose with MATMUL and its actual original-typed gradients.
  Selected materializing RESHAPE reuses exact_move_unary_0066 and an ordinary distinct target;
  physical source addressing is authenticated once. No consumer copy, shader, fallback or broad
  storage relaxation is added. That composition advanced identity to38; Conv3d image-fold
  identity advanced to39, followed by the Pool3d identity40 addition above. ABI7/schema20 remain unchanged.
- Engine rejects synchronous self-close from an admitted calling thread before shutdown/wait;
  cleanup-owner recursion stays idempotent and other closers retain completion/failure replay.
  Runtime/Metal lease tokens allocate before count publication, preventing phantom ownership on OOM.
- Metal's existing Trace dependency is exported on its Java API classpath so public observer
  lambdas compile without a test-only or consumer-reflection workaround.
- Native Metal host upload/download now drain Foundation autorelease temporaries at their own
  boundaries, including mapped exception exits. Actual raw-copy and mixed public workflows show
  stable IOAccelerator allocation categories without a Java caller-pool workaround.
- Installed API examples now include a complete runnable CPU class and correct standard-CPU FFM
  prerequisites for optional OpenBLAS discovery. The changing-length example uses the existing
  detached-value shape accessor; minimal Engine/NN Maven and standalone Gradle consumers exercise
  the documented lifecycle without checkout dependencies or an unrequested incubator module.

### Compatibility

- Sources needing neither length mode nor Pool3d retain byte-identical smallest v1-v4 encoding.
  V5 adds outer opcode33 without changing old Scan records or its pre-Pool3d constructor/dtype
  admission. Pool3d in either scope selects v6; v4/v5 still reject opcodes34/35.
  Existing fixed recurrent sources remain fixed, including old v3 files. Only new explicit F32/F64
  rnnScan recipes use general Scan; ordinary .rnn, direct fixed, GRU/LSTM and low source construction
  remain fixed. CPU low numerical availability independently follows the temporary lock above.
- Advanced ScanAttrs callers migrate from the removed nine-argument row sentinel to explicit
  ten-component lengthAware metadata. The seven-argument all-valid constructor and qualified
  old CPU IR/identity/class bytes and v1-v5 wire domains are unchanged.
