# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0049: Length-aware whole-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [general Scan](../../glossary.md#general-differentiable-scan)

## Status

Accepted — 2026-10-24. User-selected design and coordinated authority under
[Model0037](../../planning/modules/model/tasks/0037-length-aware-general-scan-authority.md) accepted
by MAIN after Hume's final overall independent Class C APPROVE, zero open P1/P2.
Implementation update — 2026-10-24: Model0038's bounded source/structural-[IR](../../glossary.md#term-intermediate-representation-ir) foundation is
implemented and independently approved. MAIN accepted Complete on 2026-10-24 after the combined
integration retry passed following a test-only ledger repair. Implementation/status amendment —
2026-10-26: Compiler0013/Prepare0011 and fixed-geometry [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) CPU0042 are Complete; Model0044
source and Compiler0015 [inference](../../glossary.md#term-inference)/actual [AD](../../glossary.md#autograd)/all-port [binding](../../glossary.md#term-binding) add one-B length [metadata](../../glossary.md#term-metadata) support.
Prepare0015's pre-Java [qualification](../../glossary.md#term-qualification) was independently approved and MAIN-accepted on 2026-10-26.
Implementation amendment — 2026-10-26: Prepare0015 implements combined one-B/fixed-T/[features](../../glossary.md#term-feature)
F32/F64 execution and its bounded outer/control/ROW(1) prerequisite. Owning [validation](../../glossary.md#term-validation) and independent
code/structural review passed; root validation and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. The original
decision's semantics, header, row/AD/activity and saved-site rules are unchanged.
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) and its incorporated contracts remain authoritative.

## Context

Existing [general Scan](../../glossary.md#general-differentiable-scan) processes all static time steps; fixed recurrent scans already have a
different length-aware [inference](../../glossary.md#term-inference) path. Training ordinary whole-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) bodies needs skipping before
padded [primal](../../glossary.md#term-primal)/[cotangent](../../glossary.md#cotangent) loads and arithmetic, not masking a result after [NaN](../../glossary.md#term-nan) has participated.
Whole-batch algebra can mix rows, so a length [Tensor](../../glossary.md#tensor) alone cannot define legal skipping.
The independently reviewed [Model0036 proposal](../../planning/length-aware-general-scan.md)
was accepted for delivery, then the user's “ok, pust se do prace” selected its design and the
immediate authority plus bounded [Model](../../glossary.md#term-model) foundation, not later [owner](../../glossary.md#term-owner) execution.

## Decision drivers

Preserve ordinary [Tensor](../../glossary.md#tensor) bodies, existing first-order formulas, exact saved sites and the single
[compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run) pipeline. Make row [identity](../../glossary.md#term-identity-meanings) explicit, preserve old all-valid behavior/bytes,
and accumulate shared [parameter](../../glossary.md#term-parameter) [gradients](../../glossary.md#term-gradient) without B×parameter state. Keep new source [validation](../../glossary.md#term-validation)
bounded and independent of [backend capability](../../glossary.md#backend-capability). Do not migrate fluent APIs incidentally.

## Options considered

- Keep all-valid only: preserves current behavior but cannot deliver the selected [padding](../../glossary.md#padding) semantics.
- Row-local formals with batch removed: makes separation structural but changes body binding,
  history geometry and shared-gradient folding; not selected.<span data-glossary-navigation="true"> ([batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters); [binding glossary](../../glossary.md#term-binding); [gradient glossary](../../glossary.md#term-gradient))</span>
- Whole-batch formals with explicit roles and closed checks: retains batched Tensor/derivative
  algebra and existing histories, while rejecting unproved cross-row dependence; selected.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [derivative glossary](../../glossary.md#term-derivative))</span>

## Decision

Adopt the [scan owner's complete protocol](../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>.
Extend existing ScanAttrs with rowCarryCount and [immutable](../../glossary.md#term-immutable) ordered List<OptionalInt> sequenceRowAxes,
and [Scan](../../glossary.md#term-cumulative-scan).apply with a final-length-actual [binding](../../glossary.md#term-binding) overload. No lengths body formal or new
production type is selected. New length-mode source admits [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) and only
the selected closed row-preserving algebra; old all-valid [PURE](../../glossary.md#term-pure)/type [admission](../../glossary.md#term-admission) is untouched.
Leading carries are ROW(0); terminal SHARED suffixes have distinct old.add(contribution) sites.
Explicit ROW(0)/ROW(1)/SHARED roles must survive saved-history pruning and later transport.

Skip inactive loads/math, preserve raw inactive [carry](../../glossary.md#term-carry) [identity](../../glossary.md#term-identity-meanings) and positive-zero output [padding](../../glossary.md#padding).
First-order original sources require every carry and every original sequence to be ROW(0),
including unused/non-[gradient](../../glossary.md#term-gradient) sequences. Generated reverse inputs may have mixed roles; existing
higher-order rejection remains. Reduce active contributors in increasing [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) order, then one
shared ADD/time; never per-row [parameter](../../glossary.md#term-parameter) banks. Full lengths [validation](../../glossary.md#term-validation) follows [producers](../../glossary.md#term-producer) and
precedes the first scan-local write. Shared work/history needs active-time admission too.

Select future [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model).v5/SYNGRF05/version5 and new outer opcode33, verified free after the
current 1..32 inventory. Keep old opcode23/seven attrs, v1–v4 bytes and old-default/fixed [CPU](../../glossary.md#term-cpu)
identity streams unchanged. Constructor bridges preserve calls, not record reflection/[hash](../../glossary.md#term-hash)/text
shape. Changed record toString must never become an old-default key encoding.

## Rationale

For x `[B,F]` @ shared W `[F,H]`, row-leading results retain ordinary body geometry. Its reverse
can reduce active contributors into one `[F,H]` sum instead of B copies. A saved shared W [transpose](../../glossary.md#term-transpose)
is `[T,F,H]`, not a row history even at F=B. Explicit roles prevent that equal-size ambiguity;
terminal-only row reductions prevent coupling from flowing back into row state. The closed
algebra is sufficient for the selected ordinary [RNN](../../glossary.md#term-recurrent-neural-network-rnn)-shaped body without a general proof system.

## Consequences

New source [API](../../glossary.md#term-application-programming-interface-api)/record shape and length-mode contributor domains require independent review and
negative tests. Histories can still cost O(T×P) shared or O(T×B×row-[elements](../../glossary.md#term-element)); demand-only storage
does not make all memory [constant](../../glossary.md#term-constant). New reduction grouping has its own [oracle](../../glossary.md#term-test-oracle), not a bitwise
equivalence promise to another grouping. Mixed [INT64](../../glossary.md#data-type--datatype)/floating [runtime](../../glossary.md#term-runtime) boundaries need real work.

Authority review/MAIN acceptance precedes [Model0038](../../planning/modules/model/tasks/0038-length-aware-scan-source-foundation.md).
That foundation itself enabled no route: at its acceptance, old [Compiler](../../glossary.md#term-compiler) [inference](../../glossary.md#term-inference)/[AD](../../glossary.md#autograd)/cold,
[Prepare](../../glossary.md#prepare), [CPU](../../glossary.md#term-cpu) [pointwise](../../glossary.md#term-pointwise)/shaped and [Checkpoint](../../glossary.md#term-checkpoint) encoder explicitly rejected length mode. The later
implemented and accepted-[target](../../glossary.md#term-target) scopes are distinguished in Status above. Exact v5 field
encoding/compatibility tests are separately gated, not a task cascade. [Dependency direction](../../glossary.md#java-platform-and-build-tools) and
execution/resource [owners](../../glossary.md#term-owner) do not change, so this documentation phase changes no architecture tests.

No fluent `.rnn`/fixed-[scan](../../glossary.md#term-cumulative-scan) migration, fixed [BPTT](../../glossary.md#term-backpropagation-through-time-bptt), [GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) migration, higher-order waiver, implicit
[model](../../glossary.md#term-model)-wide [mask](../../glossary.md#term-boolean-mask)/[MEAN](../../glossary.md#term-mean) change, packing, new executor, low/[Metal](../../glossary.md#term-metal)/[vector](../../glossary.md#term-vector)/parallel/[OpenBLAS](../../glossary.md#term-openblas) route.
NN0032's advanced second-order CPU [compile](../../glossary.md#compile) [regression](../../glossary.md#term-regression), [singleton](../../glossary.md#term-singleton)-CONCAT copy and unproved
mixed-[engine](../../glossary.md#term-engine) compatibility remain separate gates. No performance/build [run](../../glossary.md#term-run) is part of this decision.

## Related documentation

The [focused explanation](../../architecture/general-differentiable-scan.md#accepted-length-aware-design-implementation-pending)
illustrates skipping and shared accumulation. [Checkpoint v5](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
owns future wire/bounds duties, not [scan](../../glossary.md#term-cumulative-scan) semantics. [ADR0045](0045-general-differentiable-scan.md)
and [ADR0047](0047-bounded-general-scan-graph-persistence.md) remain valid for the unchanged all-valid baseline.
