# Changelog

## 0.1.0 — private/internal

### Added

- [Bounded CPU autotuning coverage and correctness](docs/planning/autotuning-audit.md#verified-cpu-coverage-and-fixes):
  truthful caller-input copies; legal scalar/vector/range and qualified OpenBLAS/oneDNN choices;
  frozen selected configurations; complete-only and singleton transactions; copy geometry,
  bias-stride, heap-ingress, zero-work and resource-policy repairs. Internal CPU identities are
  versioned. Actual public/native runs and affected tests prove the reached paths; exact floating
  disagreement still fails strict comparison, and architecture-changing expansions are proposals.

- [Autotuning transaction repairs](docs/planning/autotuning-audit.md#reproduced-and-verified-transaction-repairs):
  accept valid `nanoTime()` rollover intervals, read each cache from one opened-file snapshot,
  and preserve primary publication failures when temporary cleanup also fails. Existing cache
  formats, checked work/size limits and Runtime ownership remain unchanged. Actual before/after
  programs and affected-suite results are recorded in the audit, separately from source review.

- [Whole-document terminology and comprehension revision](docs/planning/documentation-terminology-validation.md):
  local caller/callee, ownership and graph-leaf explanations; a prerequisite-linked dictionary;
  LaTeX arithmetic/calculus; why-comments and clearer sections; focused related-topic links and a
  revised GitHub README. Corrected wrong-sense glossary navigation, factual execution/persistence
  boundaries and recurrent parameter packing. Actual rendering, JDK token preservation and copied
  program compilation/execution are recorded separately from model-based review. Runtime source,
  architecture, old qualification artifacts and private-distribution rules are unchanged.

- Reader-oriented documentation now separates tutorials, how-to, explanation and reference.
  One complete seven-stage FLOAT32 Java project runs against installed private artifacts;
  the source-grounded atlas explains all 78 Java packages and 43 native code/build components.
  Focused reference pages retain all 542 original incoming anchors, archive superseded inventories
  and correct stale Scan/BPTT/normalization claims. Actual examples, Javadoc, reader navigation and
  light/dark diagrams are qualified separately from automated reader reviews; no public licence
  or new backend capability is implied.<span data-glossary-navigation="true"> ([FLOAT32 glossary](docs/glossary.md#term-float32); [artifacts glossary](docs/glossary.md#term-artifact); [native code glossary](docs/glossary.md#native-numerical-library-and-instruction-architecture); [BPTT glossary](docs/glossary.md#term-backpropagation-through-time-bptt); [normalization glossary](docs/glossary.md#term-normalization); [Javadoc glossary](docs/glossary.md#term-javadoc); [backend capability glossary](docs/glossary.md#backend-capability))</span>

- Public-readiness stabilization preserves full-precision logical/master restore identity,
  bounded linear namespace acquisition, session-only cache isolation and exact tuning-batch
  rebinding; caller-storage prefixes, partial resource adoption and terminal shutdown retain
  ownership and original failure semantics.<span data-glossary-navigation="true"> ([precision glossary](docs/glossary.md#term-precision); [identity glossary](docs/glossary.md#term-identity-meanings); [linear glossary](docs/glossary.md#term-linear-meanings); [cache glossary](docs/glossary.md#term-cache); [batch glossary](docs/glossary.md#term-batch-machine-learning-and-hyperparameters); [ownership glossary](docs/glossary.md#term-ownership-meanings))</span>
- Corrected CPU statistical working sites, square-overflow classes, L2 point identities,
  exact pooling/attention divisors and categorical-loss gradients. Selective compatibility
  projections 79/80 retire affected generated artifacts without bumping unrelated raw families.<span data-glossary-navigation="true"> ([overflow glossary](docs/glossary.md#term-overflow); [L2 glossary](docs/glossary.md#term-l2); [pooling glossary](docs/glossary.md#term-pooling); [loss glossary](docs/glossary.md#loss); [gradients glossary](docs/glossary.md#term-gradient))</span>
- Corrected Metal Conv2d/Conv3d contraction-before-bias ordering, active source-only BOOL Scan
  validation, typed storage/control admission and empty-context lifetime checks. Scan grammar v2
  declares a run-private first-failure record; coordinated identity45 and renewed shader
  authentication reject stale layouts and decisions. Native causes remain attached to public errors.<span data-glossary-navigation="true"> ([Metal glossary](docs/glossary.md#term-metal); [Conv2d glossary](docs/glossary.md#convolution--conv2d); [Conv3d glossary](docs/glossary.md#conv3d--ncdhw); [contraction glossary](docs/glossary.md#contraction-dimension); [bias glossary](docs/glossary.md#term-bias); [BOOL glossary](docs/glossary.md#data-type--datatype); [validation glossary](docs/glossary.md#term-validation); [admission glossary](docs/glossary.md#term-admission); [lifetime glossary](docs/glossary.md#term-lifetime); [run glossary](docs/glossary.md#term-run); [shader glossary](docs/glossary.md#term-shader); [authentication glossary](docs/glossary.md#term-authentication); [layouts glossary](docs/glossary.md#layout))</span>
- Optional CPU preparation and ordinary Engine tuning observers report bounded detached facts
  through the existing Trace envelope. Unobserved overloads perform no diagnostic allocation or
  clock work; native execute/wait preserves the primary failure with later wait failure suppressed.<span data-glossary-navigation="true"> ([Engine glossary](docs/glossary.md#term-engine); [detached glossary](docs/glossary.md#detached-model-results); [Trace glossary](docs/glossary.md#trace))</span>

- [CPU](docs/glossary.md#term-cpu) [BFLOAT16](docs/glossary.md#term-bfloat16)/[FLOAT16](docs/glossary.md#term-float16) numerical execution and [OFP8](docs/glossary.md#term-ofp8) `FLOAT8_E4M3`/`FLOAT8_E5M2` support through
  the existing operation, shared Compiler AD, Scan/recurrent, symbolic-batch and Training owners.
  Low arithmetic uses FLOAT32 work with one final narrowing; FP8 storage, selection and host
  publication preserve raw byte bits. Explicit conversions round directly to the target.
  Training retains FLOAT32 masters and rejects nonfinite logical candidates transactionally.
  Conditional graph v7 and NN model/definition/whole-model v2 add FP8 without changing old
  non-FP8 bytes or promising optimizer continuation.<span data-glossary-navigation="true"> ([operation glossary](docs/glossary.md#operation); [Compiler glossary](docs/glossary.md#term-compiler); [AD glossary](docs/glossary.md#autograd); [narrowing glossary](docs/glossary.md#term-narrowing); [FP8 glossary](docs/glossary.md#term-fp8); [host glossary](docs/glossary.md#term-host); [publication glossary](docs/glossary.md#term-publication); [target glossary](docs/glossary.md#term-target); [NN model glossary](docs/glossary.md#model-module-buffer-and-scan-distinctions); [model glossary](docs/glossary.md#term-model); [optimizer glossary](docs/glossary.md#term-optimizer))</span>
- Typed oneDNN low-input MATMUL with strict FLOAT32 output staging and generated logical narrowing,
  independently qualified per API/implementation/target. Arm FP8 stays on the ordinary CPU path;
  no widened FLOAT32 input is advertised as native FP8. F64 oneDNN expansion remains excluded.<span data-glossary-navigation="true"> ([MATMUL glossary](docs/glossary.md#matrix-multiplication--matmul); [API glossary](docs/glossary.md#term-application-programming-interface-api); [F64 glossary](docs/glossary.md#floating-point-precision-and-rounding))</span>

- Explicit [oneDNN](docs/glossary.md#term-onednn) 3.14.0 [LP64](docs/glossary.md#term-lp64) sequential CPU composition with `openOneDnn(Path)` and [immutable](docs/glossary.md#term-immutable)
  `AUTO`/`PRIMITIVES`/[`GRAPH`](docs/glossary.md#graph) API preferences inside the existing CPU owner. Required explicit
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
  baselines are recorded in the <span data-glossary-navigation="true"> ([owner glossary](docs/glossary.md#term-owner); [OpenBLAS glossary](docs/glossary.md#term-openblas); [binary glossary](docs/glossary.md#term-binary-descriptor-and-metadata); [homogeneous glossary](docs/glossary.md#term-homogeneous); [rank glossary](docs/glossary.md#term-rank); [F32 glossary](docs/glossary.md#floating-point-precision-and-rounding); [MSE glossary](docs/glossary.md#mean-squared-error--mse); [opaque glossary](docs/glossary.md#term-opaque); [convolution glossary](docs/glossary.md#term-channels-first-convolution-layers); [activation glossary](docs/glossary.md#term-activation); [mean glossary](docs/glossary.md#term-mean); [Singleton glossary](docs/glossary.md#term-singleton); [stride glossary](docs/glossary.md#term-stride); [aliasing glossary](docs/glossary.md#term-aliasing); [Shape glossary](docs/glossary.md#shape); [backend glossary](docs/glossary.md#backend); [descriptor glossary](docs/glossary.md#term-descriptor); [IR glossary](docs/glossary.md#term-intermediate-representation-ir); [heuristic glossary](docs/glossary.md#term-heuristic); [weight glossary](docs/glossary.md#term-weight); [schema glossary](docs/glossary.md#term-schema); [arm64 glossary](docs/glossary.md#native-packaging-and-platform-vocabulary); [x86-64 glossary](docs/glossary.md#term-x86-64); [qualification glossary](docs/glossary.md#term-qualification))</span>[configuration and exact binary pins](docs/backend-guide/cpu-backend.md#explicit-onednn-composition).

- Qualified shared Metal FLOAT32/BFLOAT16/FLOAT16 elementary, scalar/tensor power,
  masked/statistical reduction and cumulative execution, including rank-zero values, affine
  scalar/clamp/reduction reads and generated expanded-one counts. Numerical bodies use FLOAT32
  working sites and original-carrier publication; cumulative INT32/INT64 arithmetic is modular.<span data-glossary-navigation="true"> ([scalar glossary](docs/glossary.md#term-scalar); [tensor glossary](docs/glossary.md#tensor); [rank-zero glossary](docs/glossary.md#axis-rank-scalar-vector-and-matrix); [carrier glossary](docs/glossary.md#term-storage-carrier); [INT32 glossary](docs/glossary.md#data-type--datatype); [INT64 glossary](docs/glossary.md#data-type--datatype))</span>
- Metal normalization, categorical losses and attention now share original-carrier FLOAT32,
  BFLOAT16 and FLOAT16 kernel bodies, with explicit per-role metadata, Model promotion and
  typed epsilon/scale bits. Attention keeps scores/probabilities in declared FLOAT32 work
  storage until each final output is narrowed; low weights siblings do not replace that work.
  Independent raw validation preserves canonical batch-statistic siblings, labels, masks and
  exact gradient role flags. FLOAT16 attention accepts its documented FLOAT32 scale metadata
  in both public construction and captured inference; CPU follows its independent low-precision cutover.<span data-glossary-navigation="true"> ([kernel glossary](docs/glossary.md#kernel); [epsilon glossary](docs/glossary.md#epsilon); [canonical glossary](docs/glossary.md#term-canonical-form); [statistic glossary](docs/glossary.md#term-statistic); [masks glossary](docs/glossary.md#term-boolean-mask); [inference glossary](docs/glossary.md#term-inference-meanings))</span>
- Removed the Metal CONCAT/STACK sixteen-operand binding limit through dynamic references and
  disjoint copies directly into final storage, shared by ordinary and controlled Scan execution.
  Actual public GPU qualification covers 17/33/257 whole inputs, last-operand-only gradients,
  empty outputs/cotangents and 33 distinct Scan operands in both directions and activity modes
  across FLOAT32/BFLOAT16/FLOAT16.<span data-glossary-navigation="true"> ([binding glossary](docs/glossary.md#term-binding); [GPU glossary](docs/glossary.md#term-gpu))</span>
- Native Metal general Scan through authenticated bounded body/binder plans and once-prepared
  ordinary kernels, including actual reverse bodies and runtime-length RNN training/save/load/
  fresh training. Structural inactivity guards operands before loads or index validation;
  there is no Java body loop, time unrolling or CPU fallback.<span data-glossary-navigation="true"> ([general Scan glossary](docs/glossary.md#general-differentiable-scan); [kernels glossary](docs/glossary.md#kernel); [runtime glossary](docs/glossary.md#term-runtime); [RNN glossary](docs/glossary.md#term-recurrent-neural-network-rnn); [unrolling glossary](docs/glossary.md#term-unrolling))</span>
- Original-typed Metal exact stored F64/integral operations and promotion, independent Model
  metadata inference, canonical multi-output sibling retention and source-only gradient
  publications through the existing initializer and per-run resource owner.<span data-glossary-navigation="true"> ([resource owner glossary](docs/glossary.md#ownership-borrowing-and-lease))</span>
- First-order fixed RNN_TANH, GRU_RESET_AFTER and LSTM gradients through the shared length-aware
  Scan/ordinary formula owners, including sequence/final-state seeds, both directions and empty
  time/batch domains. Fixed Metal forward uses the common F32/BF16/F16 numerical codec with
  F32 low-carrier working sites and explicit recurrent-state narrowing. CPU low arithmetic
  admission and fixed recurrent higher-order restrictions remain unchanged.<span data-glossary-navigation="true"> ([LSTM glossary](docs/glossary.md#term-long-short-term-memory-lstm); [seeds glossary](docs/glossary.md#commit-release-distribution-log-and-seed-meanings); [BF16 glossary](docs/glossary.md#floating-point-precision-and-rounding); [F16 glossary](docs/glossary.md#floating-point-precision-and-rounding); [codec glossary](docs/glossary.md#term-codec))</span>

- Model-valid CPU promoted F32/F64 and I32/I64 pointwise operands, exact comparisons and selected
  WHERE conversion, retaining each gradient target's original carrier.<span data-glossary-navigation="true"> ([pointwise glossary](docs/glossary.md#term-pointwise))</span>
- Checked CPU signed/empty slice materialization and its existing extraction/update pullbacks,
  bounded one-pass variadic CONCAT/STACK beyond sixteen inputs, and shared sound output
  injectivity proofs for large interleaved layouts.<span data-glossary-navigation="true"> ([slice glossary](docs/glossary.md#slice); [materialization glossary](docs/glossary.md#materialization); [variadic glossary](docs/glossary.md#term-variadic); [injectivity glossary](docs/glossary.md#storage-mapping-contiguity-and-injectivity))</span>

- Shared Metal FLOAT32/BFLOAT16/FLOAT16 matmul, convolution, pooling and overlapping
  axis/image fold execution owners, with FLOAT32 working sites and one final low-format
  conversion after actual additions; single-contributor fold sites preserve original raw words.
  General scatter reductions and Gather cotangents retain original value carriers,
  multidimensional indices and gradient flags. Raw signed slices materialize all seven carriers;
  mixed floating comparisons use exact represented-value ordering. Metal FLOAT64 folding remains
  non-overlapping and CPU BFLOAT16/FLOAT16 arithmetic restrictions remain unchanged.<span data-glossary-navigation="true"> ([axis glossary](docs/glossary.md#term-axis); [Gather glossary](docs/glossary.md#gather); [FLOAT64 glossary](docs/glossary.md#term-float64))</span>
- Fluent `ModelBuilder.maxPool3d(kd,kh,kw)` and `averagePool3d(kd,kh,kw)`, with kernel stride,
  zero padding, unit dilation and floor geometry; ordinary Conv3d/pooling/linear Model training
  and save/load continue through existing owners with freshly configured training.<span data-glossary-navigation="true"> ([padding glossary](docs/glossary.md#padding); [dilation glossary](docs/glossary.md#term-dilation))</span>
- Complete typed Pool3d cold geometry, including otherwise-qualified all-valid and static
  homogeneous FLOAT32/FLOAT64 length-aware Scan bodies. Only these two constructors are added
  to the narrow length whitelist; existing role, axis and terminal restrictions remain.<span data-glossary-navigation="true"> ([Pool3d glossary](docs/glossary.md#pool3d--ncdhw-pooling))</span>
- Append-only `graph-model.v6` (`SYNGRF06`, version 6), opcodes 34/35 for maximum/average Pool3d,
  twelve i64 geometry fields and a canonical ceil flag. Contained Pool3d also selects v6;
  unchanged sources keep byte-identical smallest v1–v5 encoding and no optimizer/session state.
- [Metadata](docs/glossary.md#term-metadata)-only `GraphCompilationPort.validateSource` shares complete cold preflight without
  Tensor identities or host copying. Decode now rejects invalid metadata before creating any
  candidate Tensor for every supported version.<span data-glossary-navigation="true"> ([preflight glossary](docs/glossary.md#term-preflight))</span>
- The Pool3d checkpoint advanced Metal identity to40 and the previous audit used identity41/
  schema21. The shared numerical cutover uses identity42/schema22/header124, retaining operation116/
  attribute42 for Scan and ABI7/thirteen exports.<span data-glossary-navigation="true"> ([checkpoint glossary](docs/glossary.md#term-checkpoint))</span>
- Static Model-legal per-role FLOAT32/FLOAT64/BFLOAT16/FLOAT16 general Scan cold capture,
  including legal casts, Checkpoint copy/rebind and NN acquisition/cold-forward paths.
  Reconstruction preserves explicit row roles, aliases and canonical ports without unrolling
  time, reading external lengths or loosening exact rebind descriptors. Symbolic cold bodies
  remain outside this domain; Metal FLOAT64 numerical execution is not implied.<span data-glossary-navigation="true"> ([capture glossary](docs/glossary.md#term-capture-meanings); [rebind glossary](docs/glossary.md#capture-graph-ports-and-reconstruction); [NN glossary](docs/glossary.md#neural-network-module-parameter-buffer-and-forward-context); [aliases glossary](docs/glossary.md#aliasing-and-arena))</span>
- Existing selective graph-model.v5/v6 persistence retains row roles, aliases, canonical ports
  and named values without storing external length batches or changing checkpoint wire versions.
  Loaded typed Models accept fresh lengths and newly configured training.<span data-glossary-navigation="true"> ([persistence glossary](docs/glossary.md#term-persistence))</span>
- Explicit `RnnSequence.forwardGeneralScan` and trainable `rnnScan` Models support all four
  floating carriers through the ordinary RnnCell equation and initialization. Existing low
  default builders retain fixed-cell working/publication semantics. Semantic final-hidden
  selection supports general/fixed ports; fresh-length training and save/close/load/new training
  use the existing execution owners.
- True empty-carry Scan source and explicit lengthAware metadata, including zero-ROW Shared
  source with B derived from lengths. All canonical typed ports/aliases remain; source validity,
  selected derivatives, backend execution and cold/wire admission remain separate checks.<span data-glossary-navigation="true"> ([carry glossary](docs/glossary.md#term-carry); [derivatives glossary](docs/glossary.md#term-derivative))</span>
- Descriptor-only ordinary checking through ScanBody.describeOrdinary, sharing existing Tensor
  family guards and all canonical siblings without allocating Tensor identities. Source checking
  does not widen length-aware row proof, AD, backend or persistence support.
- Versioned local Maven, binary and exact-source distributions for eighteen implemented products,
  with source/Javadoc JARs, real public dependency closure, third-party notices, explicit source
  provenance and reproducible source archives. External Maven/Gradle CPU/Metal model consumers
  run without checkout JARs. The bounded private/internal 0.1.0 matrix is technically qualified
  through the root gate, isolated Maven replay and clean eighteen-product NoGit/source-native
  rebuilds. Public project licensing, registry upload and trusted native-prebuilt signing are separate.<span data-glossary-navigation="true"> ([Maven glossary](docs/glossary.md#term-maven); [distributions glossary](docs/glossary.md#commit-release-distribution-log-and-seed-meanings); [dependency glossary](docs/glossary.md#term-dependency); [source provenance glossary](docs/glossary.md#term-source-provenance); [Gradle glossary](docs/glossary.md#term-gradle); [matrix glossary](docs/glossary.md#term-matrix-meanings); [registry glossary](docs/glossary.md#term-registry))</span>
- Shared per-output ordinary Scan coordinate metadata, including canonical BatchNorm/attention
  dependencies, completely proved nonleading intermediate rows and terminal contributor routing.
  Model/raw forward validation reuse one seam; retained AD/header/physical/cold/wire boundaries
  remain independent and unrepresentable saved histories reject before derivative identities.<span data-glossary-navigation="true"> ([coordinate glossary](docs/glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- General CPU Scan composition through existing ordinary lowerers and emitters, for all-valid and
  length-aware static/qualified one-B bodies with per-value types, exact selected first-order VJPs,
  structural absence distinct from present zero and run-private physical/control storage.
  Genuine qualified legacy keys, ABIs and generated bytes remain replayable. This adds no second
  executor, broader cold/wire format, Metal Scan capability or Java low-arithmetic unlock.<span data-glossary-navigation="true"> ([wire format glossary](docs/glossary.md#term-wire-format))</span>
- Trainable grouped Conv3d through shared Conv2d/Conv3d unfold/MATMUL/overlap-add-fold adjoints:
  CPU FLOAT32/FLOAT64 and Metal FLOAT32 have every-input/weight/bias finite-difference proof,
  including biased grouped anisotropic and unbiased depthwise cases, plus real NN training,
  bit-exact parameter save/close/load and newly configured training. CPU general Scan composes
  the complete Conv3d forward/reverse body in both types, modes and traversal directions.
  At that checkpoint CPU Pool3d required no-gradient occurrences; the Pool3d addition above
  supersedes that gate. Ordinary public CPU named-B Conv3d, CPU low arithmetic and Metal Scan
  are not widened by either backend change.<span data-glossary-navigation="true"> ([finite-difference glossary](docs/glossary.md#term-finite-difference-gradient-check); [parameter glossary](docs/glossary.md#term-parameter))</span>

### Changed

- Live CPU/Metal audits cover all 115 ordinary operation kinds, 137 signatures and separate Scan,
  classifying execution, selection, illegal projections and nondifferentiable/deferred roles.
  Each Metal F32/BF16/F16 family executes the same 187 forward and 358 selected-gradient cases
  without a Model-legal selection gap. Execution counts do not substitute for independent
  numerical and public model/training/checkpoint checks.<span data-glossary-navigation="true"> ([CPU glossary](docs/glossary.md#term-cpu); [Metal glossary](docs/glossary.md#term-metal); [operation glossary](docs/glossary.md#operation); [signatures glossary](docs/glossary.md#term-signature); [F32 glossary](docs/glossary.md#floating-point-precision-and-rounding); [BF16 glossary](docs/glossary.md#floating-point-precision-and-rounding); [F16 glossary](docs/glossary.md#floating-point-precision-and-rounding); [gradient glossary](docs/glossary.md#term-gradient); [Model glossary](docs/glossary.md#term-model); [checkpoint glossary](docs/glossary.md#term-checkpoint))</span>
- One Metal fixed-corpus authentication covers 23 components/187,750 UTF-8 bytes and the common
  original-carrier codec; the separate low library and frozen FLOAT32 descriptor proxy are retired.
  General FLOAT64 arithmetic and CPU low arithmetic are not added.<span data-glossary-navigation="true"> ([authentication glossary](docs/glossary.md#term-authentication); [UTF-8 glossary](docs/glossary.md#term-utf-8); [carrier glossary](docs/glossary.md#term-storage-carrier); [codec glossary](docs/glossary.md#term-codec); [FLOAT32 glossary](docs/glossary.md#term-float32); [descriptor glossary](docs/glossary.md#term-descriptor); [FLOAT64 glossary](docs/glossary.md#term-float64))</span>
- Temporarily lock Java CPU BFLOAT16/FLOAT16 numerical computation at source, preparation,
  represented-IR, direct generation/definition, retained cache and fixed-recurrent admission.
  Keep the implementation, exact/raw/discrete/storage/conversion roles and explicit F32 compute/
  F32-master training. Metal low support, shared types and old qualified generated identities remain.
  No runtime unlock, hidden low emulation or blanket low-containing-partition rejection is added.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](docs/glossary.md#term-bfloat16); [FLOAT16 glossary](docs/glossary.md#term-float16); [IR glossary](docs/glossary.md#term-intermediate-representation-ir); [cache glossary](docs/glossary.md#term-cache); [admission glossary](docs/glossary.md#term-admission); [runtime glossary](docs/glossary.md#term-runtime); [partition glossary](docs/glossary.md#partition))</span>
- Ordinary CPU generation shares one namespaced method-emission dispatch for standalone classes
  and same-class helpers, including existing loss and row-phase methods. Standalone generated
  identities and class bytes remain unchanged; this factoring does not widen Scan execution.<span data-glossary-navigation="true"> ([dispatch glossary](docs/glossary.md#term-dispatch); [loss glossary](docs/glossary.md#loss))</span>
- All eleven ordinary derivative families share one formula body for direct Tensor construction
  and identity-free checked metadata construction. Model guards and independent Compiler inference
  remain distinct; exact constants, lexical primal bindings and optional eager-site observation
  are preserved. This does not itself widen Scan admission or add a derivative implementation.<span data-glossary-navigation="true"> ([derivative glossary](docs/glossary.md#term-derivative); [Tensor glossary](docs/glossary.md#tensor); [identity glossary](docs/glossary.md#term-identity-meanings); [metadata glossary](docs/glossary.md#term-metadata); [Compiler glossary](docs/glossary.md#term-compiler); [inference glossary](docs/glossary.md#term-inference-meanings); [constants glossary](docs/glossary.md#term-constant); [primal glossary](docs/glossary.md#term-primal); [bindings glossary](docs/glossary.md#term-binding); [eager glossary](docs/glossary.md#term-eager-evaluation))</span>
- Compiler and CPU local projection share ordinary computed logical-layout completion on Model
  ScanBody. Existing affine/crop geometry, materializing-RESHAPE decisions and the separate generic
  symbolic classifier have one owner; captured graphs, ordinary inference and strict ingress
  guards remain unchanged.<span data-glossary-navigation="true"> ([layout glossary](docs/glossary.md#layout); [owner glossary](docs/glossary.md#term-owner); [graphs glossary](docs/glossary.md#graph))</span>
- CPU generated raw F32/F64/BF16/F16 UNFOLD3D and arithmetic F32/F64 FOLD3D share existing
  movement/fold owners, preserving qualified legacy identities and the low arithmetic lock.<span data-glossary-navigation="true"> ([F64 glossary](docs/glossary.md#floating-point-precision-and-rounding))</span>

### Fixed

- CPU FLOAT32 RSQRT preserves its square-root working site before reciprocal across scalar,
  vector/remainder and composed helpers; selective schema 76 rejects stale affected bodies.<span data-glossary-navigation="true"> ([CPU glossary](docs/glossary.md#term-cpu); [FLOAT32 glossary](docs/glossary.md#term-float32); [scalar glossary](docs/glossary.md#term-scalar); [vector glossary](docs/glossary.md#term-vector); [schema glossary](docs/glossary.md#term-schema))</span>
- CPU dropout now uses typed complement/reciprocal/multiply with numeric identity v2, preserving
  exact RNG words, masks and next-state. SOFTMAX/LOG_SOFTMAX algorithm 3 handles the Model's
  nonfinite and empty-slice domain without a separate finite-only validator.<span data-glossary-navigation="true"> ([dropout glossary](docs/glossary.md#dropout--inverted-dropout); [identity glossary](docs/glossary.md#term-identity-meanings); [RNG glossary](docs/glossary.md#graph-rng-state--graphrngstate); [masks glossary](docs/glossary.md#term-boolean-mask); [SOFTMAX glossary](docs/glossary.md#softmax--log-softmax); [Model glossary](docs/glossary.md#term-model); [slice glossary](docs/glossary.md#slice); [finite glossary](docs/glossary.md#term-finite))</span>
- CPU attention preserves zero-plus-singleton batch broadcasting. Wide CONCAT/STACK uses a
  prepared operand table and checked geometry under selective schema 75 instead of an N-wide
  JVM entry budget; small existing entries and controlled arena ownership remain unchanged.<span data-glossary-navigation="true"> ([singleton glossary](docs/glossary.md#term-singleton); [batch glossary](docs/glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](docs/glossary.md#broadcasting); [JVM glossary](docs/glossary.md#term-java-virtual-machine-jvm); [arena glossary](docs/glossary.md#term-arena); [ownership glossary](docs/glossary.md#term-ownership-meanings))</span>
- Metal semantic identity 44 aligns affine unary/comparison/MATMUL capability, preparation and
  raw preflight, closes ordinary empty and zero-grid fused-anchor execution, and maps proved
  numeric Scan ingress through existing CONTIGUOUS. Canonical non-gradient raw F64 Scan stays
  source-only; ABI7/schema22 and intentional numerical/resource restrictions remain.<span data-glossary-navigation="true"> ([Metal glossary](docs/glossary.md#term-metal); [unary glossary](docs/glossary.md#term-unary); [MATMUL glossary](docs/glossary.md#matrix-multiplication--matmul); [preflight glossary](docs/glossary.md#term-preflight); [fused glossary](docs/glossary.md#specialization-and-fusion); [CONTIGUOUS glossary](docs/glossary.md#term-contiguous); [Canonical glossary](docs/glossary.md#term-canonical-form); [gradient glossary](docs/glossary.md#term-gradient); [F64 glossary](docs/glossary.md#floating-point-precision-and-rounding))</span>
- Cold carry-free and zero-ROW/scalar-SHARED Scan reconstruction/persistence uses the existing
  v4/v5/v6 records, deriving batch geometry from the external lengths descriptor.<span data-glossary-navigation="true"> ([carry glossary](docs/glossary.md#term-carry); [persistence glossary](docs/glossary.md#term-persistence); [descriptor glossary](docs/glossary.md#term-descriptor))</span>
- Owned fixed and mixed recurrent Models now train through the existing TrainingSession and
  Compiler first-order AD, retaining original recipes/carriers and fresh-training-after-load
  semantics. CPU low arithmetic and higher-order recurrence are not unlocked.<span data-glossary-navigation="true"> ([TrainingSession glossary](docs/glossary.md#training-session--trainingsession); [Compiler glossary](docs/glossary.md#term-compiler); [AD glossary](docs/glossary.md#autograd))</span>
- CPU qualification no longer requires macOS temporary paths, a particular vector width,
  generated-code hashes or latency thresholds. Numerical controls retain exact non-NaN
  comparisons, typed overflow, subrange/canary checks and the selected StrictMath TANH
  producing site; arithmetic NaNs are checked by class rather than unspecified payload bits.<span data-glossary-navigation="true"> ([qualification glossary](docs/glossary.md#term-qualification); [latency glossary](docs/glossary.md#term-latency); [NaN glossary](docs/glossary.md#term-nan); [overflow glossary](docs/glossary.md#term-overflow); [canary glossary](docs/glossary.md#term-canary); [TANH glossary](docs/glossary.md#term-tanh); [payload glossary](docs/glossary.md#artifact-envelope-and-atomic-publication))</span>
- CPU one-term MATMUL preserves its exact FLOAT32/FLOAT64 product in scalar, vector, tiled,
  ordinary-batch and controlled/regional Scan execution. Empty/inactive sums retain positive
  zero and multi-term typed sites remain unchanged. Selective artifact identity rejects stale
  contraction bodies; unqualified OpenBLAS singleton contractions remain on the planned CPU path.
  Numerical regressions replace obsolete literal ClassFile/opcode snapshots.<span data-glossary-navigation="true"> ([FLOAT64 glossary](docs/glossary.md#term-float64); [positive zero glossary](docs/glossary.md#nan-infinity-and-signed-zero); [artifact glossary](docs/glossary.md#term-artifact); [contraction glossary](docs/glossary.md#contraction-dimension); [OpenBLAS glossary](docs/glossary.md#term-openblas); [opcode glossary](docs/glossary.md#term-opcode))</span>
- Metal one-term MATMUL preserves the typed product's signed zero without an added zero or
  FMA, across FLOAT32/BFLOAT16/FLOAT16. The shared custom owner guards both operands before
  loads; the existing rank-two graph owner uses exact singleton broadcast multiplication.
  Semantic identity advances to43; ABI7, program schema22 and multi-term arithmetic stay unchanged.<span data-glossary-navigation="true"> ([signed zero glossary](docs/glossary.md#term-signed-zero); [FMA glossary](docs/glossary.md#term-fused-multiply-add-fma); [BFLOAT16 glossary](docs/glossary.md#term-bfloat16); [FLOAT16 glossary](docs/glossary.md#term-float16); [owner glossary](docs/glossary.md#term-owner); [rank-two glossary](docs/glossary.md#axis-rank-scalar-vector-and-matrix); [graph glossary](docs/glossary.md#graph))</span>
- Metal signed SLICE materializes all seven stored carriers without negative physical strides;
  numerical extraction/update pullbacks use existing owners. Overlapping floating FoldAxis works
  in controlled Scan rows and guards inactive NaN contributors.<span data-glossary-navigation="true"> ([strides glossary](docs/glossary.md#term-stride))</span>
- Genuine empty storage/EXPAND detects zero extents before product/stride checks, including
  zero-grid native execution without a load or dispatch. Scalar values remain one stored element.<span data-glossary-navigation="true"> ([dispatch glossary](docs/glossary.md#term-dispatch); [element glossary](docs/glossary.md#term-element))</span>
- Scalar native graph arithmetic restores declared rank after literal broadcasting; original
  FLOAT32 FLOOR/CEIL/SIGN admission agrees with native preflight and existing exact kernels.<span data-glossary-navigation="true"> ([rank glossary](docs/glossary.md#term-rank); [admission glossary](docs/glossary.md#term-admission); [kernels glossary](docs/glossary.md#kernel))</span>
- FLOAT16 normalization inference accepts the Model's FLOAT32 epsilon/momentum metadata;
  its existing normalization adjoints use FLOAT32 working expressions and publish original
  target-carrier gradients. BFLOAT16/FLOAT32/FLOAT64 metadata matching remains exact.<span data-glossary-navigation="true"> ([normalization glossary](docs/glossary.md#term-normalization); [inference glossary](docs/glossary.md#term-inference); [epsilon glossary](docs/glossary.md#epsilon); [momentum glossary](docs/glossary.md#term-momentum); [metadata glossary](docs/glossary.md#term-metadata); [target glossary](docs/glossary.md#term-target); [carrier glossary](docs/glossary.md#term-storage-carrier))</span>
- Dropout capability checks original F32/BF16/F16 value/output descriptors while preserving
  canonical BOOL mask and INT64 state roles.<span data-glossary-navigation="true"> ([F32 glossary](docs/glossary.md#floating-point-precision-and-rounding); [BF16 glossary](docs/glossary.md#floating-point-precision-and-rounding); [F16 glossary](docs/glossary.md#floating-point-precision-and-rounding); [BOOL glossary](docs/glossary.md#data-type--datatype); [INT64 glossary](docs/glossary.md#data-type--datatype))</span>
- Host snapshots resolve canonical geometry for fully static empty publications with no logical
  layout hint, preserving selected-backend ownership, leases and byte budgets. Nonempty unresolved
  publications remain rejected.<span data-glossary-navigation="true"> ([Host glossary](docs/glossary.md#term-host); [layout glossary](docs/glossary.md#layout); [backend ownership glossary](docs/glossary.md#backend-ownership); [leases glossary](docs/glossary.md#term-lease))</span>
- Metal ordinary transport checks words/spans and ordering/reduction/fold axes before inference.
  Malformed FoldAxis target axes reject deliberately before indexing the declared shape.<span data-glossary-navigation="true"> ([axes glossary](docs/glossary.md#term-axis); [shape glossary](docs/glossary.md#shape))</span>
- FLOAT32 arithmetic over affine operands now uses existing physical-stride kernels rather than
  packed MPSGraph feed binding. Offset extraction cotangents retain every contribution when a
  target is repeated in STACK, without new math or an extra copy; canonical routes are unchanged.<span data-glossary-navigation="true"> ([MPSGraph glossary](docs/glossary.md#term-mpsgraph); [binding glossary](docs/glossary.md#term-binding))</span>
- CPU unit contraction now proves convexity on the current quotient graph, preventing cyclic
  prepared schedules after earlier contractions. Oversized variadic candidate inventories use
  existing direct decomposition without relaxing fusion or generated-method resource limits.<span data-glossary-navigation="true"> ([variadic glossary](docs/glossary.md#term-variadic); [decomposition glossary](docs/glossary.md#term-decomposition); [fusion glossary](docs/glossary.md#term-fusion))</span>
- Empty affine generated ranges guard before address/carrier reads; selective projection 73
  invalidates affected old dense-array classes without advancing generator envelope 71.<span data-glossary-navigation="true"> ([guard glossary](docs/glossary.md#term-guard))</span>
- Target-relative crop inference retains raw Model descriptors until physical layout completion,
  so actual extraction adjoints accept their original output contract.<span data-glossary-navigation="true"> ([Target-relative crop glossary](docs/glossary.md#target-relative-crop))</span>
- CPU attention follows literal eligible-score EXP normalization rather than shifted softmax.
  Its nonempty weighted-value contractions and fixed-recurrent projections seed their first
  product, preserving negative zero; empty/exclusion guards remain. Attention projection 74 rejects
  obsolete shifted classes and binds the new working-site marker in both identity streams.<span data-glossary-navigation="true"> ([EXP glossary](docs/glossary.md#term-expx); [seed glossary](docs/glossary.md#term-seed); [negative zero glossary](docs/glossary.md#nan-infinity-and-signed-zero))</span>
- Current generated-code structural oracles consume live provider/preparer/generator facts instead
  of golden class-hash and row-count snapshots. Typed ABI, direct-loop and paired-Java checks remain,
  including avoidable-dispatch guards; historical checkpoint data is unchanged.<span data-glossary-navigation="true"> ([hash glossary](docs/glossary.md#term-hash); [ABI glossary](docs/glossary.md#term-application-binary-interface-abi); [checkpoint glossary](docs/glossary.md#term-checkpoint))</span>

- General Scan preserves rank-17 temporal containers around rank-16 ordinary step values.
  The ordinary rank limit applies to every local value; exact outer role binding retains the
  single time axis without widening ordinary numerical ranks.<span data-glossary-navigation="true"> ([General Scan glossary](docs/glossary.md#general-differentiable-scan); [widening glossary](docs/glossary.md#term-widening))</span>
- Default scalar CPU preparation and MATMUL discovery no longer link an unrequested incubator
  Vector API module. Available vector alternatives remain intact when the module is resolved;
  explicit vector requests keep their existing policy. Scalar kernels and identities are unchanged.<span data-glossary-navigation="true"> ([Vector API glossary](docs/glossary.md#term-vector-api); [module glossary](docs/glossary.md#term-module))</span>
- CPU FLOAT64 SOFTMAX/LOG_SOFTMAX now accepts all finite represented inputs, including internal
  negative-infinity maximum shifts, matching FLOAT32 and the existing stable Model formula.
  Nonfinite inputs still reject before consumer writes; arithmetic, saved-output derivatives,
  loss/attention formulas and generated/cache identities are unchanged.<span data-glossary-navigation="true"> ([infinity glossary](docs/glossary.md#term-infinity); [derivatives glossary](docs/glossary.md#term-derivative); [loss glossary](docs/glossary.md#loss); [cache glossary](docs/glossary.md#term-cache))</span>
- Composite Scan emits the correct interface-static MemorySegment.copy reference and advances
  only composite cache identities so trusted pre-fix envelopes cannot restore the invalid call.
  Qualified non-composite keys, entry ABIs and generated bytes remain unchanged.<span data-glossary-navigation="true"> ([MemorySegment glossary](docs/glossary.md#term-memorysegment))</span>
- Ordinary aggregate, advanced/masked reduction and loss geometry compare retained primitive
  arrays by value. Repeated preparation and live artifact reuse retain exact guards instead of
  rejecting equivalent independently lowered physical facts.<span data-glossary-navigation="true"> ([masked reduction glossary](docs/glossary.md#masked-reduction))</span>
- Complete graph preparation declares producerless published caller values through the existing
  source-resource path, including direct caller-seed gradients at T=0. Caller slots retain Compiler
  order and borrowed lifetime; constants retain existing initialization. Source-only named B,
  including zero-byte templates, coexists with genuinely fixed CPU work without changing its kernel
  geometry or relaxing Runtime defaults. Raw seven-type publication remains separate from arithmetic.<span data-glossary-navigation="true"> ([graph preparation glossary](docs/glossary.md#graph-preparation--graphpreparation); [borrowed glossary](docs/glossary.md#ownership-borrowing-and-lease); [lifetime glossary](docs/glossary.md#term-lifetime); [constants glossary](docs/glossary.md#term-constant); [kernel glossary](docs/glossary.md#kernel); [Runtime glossary](docs/glossary.md#term-runtime); [publication glossary](docs/glossary.md#term-publication))</span>
- Advanced Engine run publication now rolls back wrapper/registration failures and always releases
  admission, including cleanup errors. Closed Engine/Runtime results and states release direct
  fresh-payload references without closing caller storage; metadata, counts/geometry and existing
  failure-replay boundaries remain unchanged.<span data-glossary-navigation="true"> ([Engine glossary](docs/glossary.md#term-engine); [run glossary](docs/glossary.md#term-run); [releases glossary](docs/glossary.md#commit-release-distribution-log-and-seed-meanings); [release glossary](docs/glossary.md#term-release))</span>
- Ordinary AD construction consistently resolves lexical value operands and saved canonical
  statistics/indices through its existing binding protocol. Reduction shares one complete ordinary
  implementation; formulas, contribution identities, eligibility and current Scan bounds remain.
- Final Compiler layout closure now distinguishes a proved RESHAPE view from an ordinary static
  materializing result after noncontiguous input. Exact views and raw source/wire inference remain
  unchanged; no derivative rewrite or consumer CONTIGUOUS is inserted. Backend admission remains separate.<span data-glossary-navigation="true"> ([view glossary](docs/glossary.md#term-view-meanings); [Backend glossary](docs/glossary.md#backend))</span>
- Java CPU ordinary static noncontiguous-input RESHAPE now uses the existing affine ordinal copy
  for canonical non-view materialization, including exact low/INT/BOOL movement. Genuine views,
  checked addressing, old qualified generated identities and the low-arithmetic lock are unchanged.<span data-glossary-navigation="true"> ([materialization glossary](docs/glossary.md#materialization))</span>
- Metal dense logical views now compose with MATMUL and its actual original-typed gradients.
  Selected materializing RESHAPE reuses exact_move_unary_0066 and an ordinary distinct target;
  physical source addressing is authenticated once. No consumer copy, shader, fallback or broad
  storage relaxation is added. That composition advanced identity to38; Conv3d image-fold
  identity advanced to39, followed by the Pool3d identity40 addition above. ABI7/schema20 remain unchanged.<span data-glossary-navigation="true"> ([shader glossary](docs/glossary.md#term-shader); [Conv3d glossary](docs/glossary.md#conv3d--ncdhw); [Pool3d glossary](docs/glossary.md#pool3d--ncdhw-pooling))</span>
- Engine rejects synchronous self-close from an admitted calling thread before shutdown/wait;
  cleanup-owner recursion stays idempotent and other closers retain completion/failure replay.
  Runtime/Metal lease tokens allocate before count publication, preventing phantom ownership on OOM.<span data-glossary-navigation="true"> ([synchronous glossary](docs/glossary.md#term-synchronous); [idempotent glossary](docs/glossary.md#term-idempotent); [tokens glossary](docs/glossary.md#term-token); [OOM glossary](docs/glossary.md#term-out-of-memory-oom))</span>
- Metal's existing Trace dependency is exported on its Java API classpath so public observer
  lambdas compile without a test-only or consumer-reflection workaround.<span data-glossary-navigation="true"> ([Trace glossary](docs/glossary.md#trace); [dependency glossary](docs/glossary.md#term-dependency); [API glossary](docs/glossary.md#term-application-programming-interface-api); [classpath glossary](docs/glossary.md#term-classpath); [observer glossary](docs/glossary.md#term-observer); [compile glossary](docs/glossary.md#compile))</span>
- Native Metal host upload/download now drain Foundation autorelease temporaries at their own
  boundaries, including mapped exception exits. Actual raw-copy and mixed public workflows show
  stable IOAccelerator allocation categories without a Java caller-pool workaround.
- Installed API examples now include a complete runnable CPU class and correct standard-CPU FFM
  prerequisites for optional OpenBLAS discovery. The changing-length example uses the existing
  detached-value shape accessor; minimal Engine/NN Maven and standalone Gradle consumers exercise
  the documented lifecycle without checkout dependencies or an unrequested incubator module.<span data-glossary-navigation="true"> ([FFM glossary](docs/glossary.md#term-ffm); [detached glossary](docs/glossary.md#detached-model-results); [NN glossary](docs/glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Maven glossary](docs/glossary.md#term-maven); [Gradle glossary](docs/glossary.md#term-gradle); [lifecycle glossary](docs/glossary.md#lifecycle); [incubator module glossary](docs/glossary.md#term-incubator-module))</span>

### Compatibility

- Sources needing neither length mode nor Pool3d retain byte-identical smallest v1-v4 encoding.
  V5 adds outer opcode33 without changing old Scan records or its pre-Pool3d constructor/dtype
  admission. Pool3d in either scope selects v6; v4/v5 still reject opcodes34/35.
  Existing fixed recurrent sources remain fixed, including old v3 files. Only new explicit F32/F64
  rnnScan recipes use general Scan; ordinary .rnn, direct fixed, GRU/LSTM and low source construction
  remain fixed. CPU low numerical availability independently follows the temporary lock above.<span data-glossary-navigation="true"> ([Pool3d glossary](docs/glossary.md#pool3d--ncdhw-pooling); [dtype glossary](docs/glossary.md#data-type--datatype); [admission glossary](docs/glossary.md#term-admission); [scope glossary](docs/glossary.md#term-scope); [F32 glossary](docs/glossary.md#floating-point-precision-and-rounding); [F64 glossary](docs/glossary.md#floating-point-precision-and-rounding); [general Scan glossary](docs/glossary.md#general-differentiable-scan); [GRU glossary](docs/glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](docs/glossary.md#term-long-short-term-memory-lstm); [CPU glossary](docs/glossary.md#term-cpu))</span>
- Advanced ScanAttrs callers migrate from the removed nine-argument row sentinel to explicit
  ten-component lengthAware metadata. The seven-argument all-valid constructor and qualified
  old CPU IR/identity/class bytes and v1-v5 wire domains are unchanged.<span data-glossary-navigation="true"> ([argument glossary](docs/glossary.md#term-argument); [metadata glossary](docs/glossary.md#term-metadata); [IR glossary](docs/glossary.md#term-intermediate-representation-ir); [identity glossary](docs/glossary.md#term-identity-meanings))</span>
