# ADR 0049: Length-aware whole-batch general Scan

## Status

Accepted — 2026-10-24. User-selected design and coordinated authority under
[Model0037](../../planning/modules/model/tasks/0037-length-aware-general-scan-authority.md) accepted
by MAIN after Hume's final overall independent Class C APPROVE, zero open P1/P2.
Implementation update — 2026-10-24: Model0038's bounded source/structural-IR foundation is
implemented and independently approved. MAIN accepted Complete on 2026-10-24 after the combined
integration retry passed following a test-only ledger repair. Implementation/status amendment —
2026-10-26: Compiler0013/Prepare0011 and fixed-geometry F32/F64 CPU0042 are Complete; Model0044
source and Compiler0015 inference/actual AD/all-port binding add one-B length metadata support.
Prepare0015's pre-Java qualification was independently approved and MAIN-accepted on 2026-10-26.
Implementation amendment — 2026-10-26: Prepare0015 implements combined one-B/fixed-T/features
F32/F64 execution and its bounded outer/control/ROW(1) prerequisite. Owning validation and independent
code/structural review passed; root validation and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. The original
decision's semantics, header, row/AD/activity and saved-site rules are unchanged.
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) and its incorporated contracts remain authoritative.

## Context

Existing general Scan processes all static time steps; fixed recurrent scans already have a
different length-aware inference path. Training ordinary whole-batch bodies needs skipping before
padded primal/cotangent loads and arithmetic, not masking a result after NaN has participated.
Whole-batch algebra can mix rows, so a length Tensor alone cannot define legal skipping.
The independently reviewed [Model0036 proposal](../../planning/length-aware-general-scan.md)
was accepted for delivery, then the user's “ok, pust se do prace” selected its design and the
immediate authority plus bounded Model foundation, not later owner execution.

## Decision drivers

Preserve ordinary Tensor bodies, existing first-order formulas, exact saved sites and the single
compile/prepare/run pipeline. Make row identity explicit, preserve old all-valid behavior/bytes,
and accumulate shared parameter gradients without B×parameter state. Keep new source validation
bounded and independent of backend capability. Do not migrate fluent APIs incidentally.

## Options considered

- Keep all-valid only: preserves current behavior but cannot deliver the selected padding semantics.
- Row-local formals with batch removed: makes separation structural but changes body binding,
  history geometry and shared-gradient folding; not selected.
- Whole-batch formals with explicit roles and closed checks: retains batched Tensor/derivative
  algebra and existing histories, while rejecting unproved cross-row dependence; selected.

## Decision

Adopt the [scan owner's complete protocol](../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan).
Extend existing ScanAttrs with rowCarryCount and immutable ordered List<OptionalInt> sequenceRowAxes,
and Scan.apply with a final-length-actual binding overload. No lengths body formal or new
production type is selected. New length-mode source admits homogeneous FLOAT32/FLOAT64 and only
the selected closed row-preserving algebra; old all-valid PURE/type admission is untouched.
Leading carries are ROW(0); terminal SHARED suffixes have distinct old.add(contribution) sites.
Explicit ROW(0)/ROW(1)/SHARED roles must survive saved-history pruning and later transport.

Skip inactive loads/math, preserve raw inactive carry identity and positive-zero output padding.
First-order original sources require every carry and every original sequence to be ROW(0),
including unused/non-gradient sequences. Generated reverse inputs may have mixed roles; existing
higher-order rejection remains. Reduce active contributors in increasing batch order, then one
shared ADD/time; never per-row parameter banks. Full lengths validation follows producers and
precedes the first scan-local write. Shared work/history needs active-time admission too.

Select future graph-model.v5/SYNGRF05/version5 and new outer opcode33, verified free after the
current 1..32 inventory. Keep old opcode23/seven attrs, v1–v4 bytes and old-default/fixed CPU
identity streams unchanged. Constructor bridges preserve calls, not record reflection/hash/text
shape. Changed record toString must never become an old-default key encoding.

## Rationale

For x `[B,F]` @ shared W `[F,H]`, row-leading results retain ordinary body geometry. Its reverse
can reduce active contributors into one `[F,H]` sum instead of B copies. A saved shared W transpose
is `[T,F,H]`, not a row history even at F=B. Explicit roles prevent that equal-size ambiguity;
terminal-only row reductions prevent coupling from flowing back into row state. The closed
algebra is sufficient for the selected ordinary RNN-shaped body without a general proof system.

## Consequences

New source API/record shape and length-mode contributor domains require independent review and
negative tests. Histories can still cost O(T×P) shared or O(T×B×row-elements); demand-only storage
does not make all memory constant. New reduction grouping has its own oracle, not a bitwise
equivalence promise to another grouping. Mixed INT64/floating runtime boundaries need real work.

Authority review/MAIN acceptance precedes [Model0038](../../planning/modules/model/tasks/0038-length-aware-scan-source-foundation.md).
That foundation itself enabled no route: at its acceptance, old Compiler inference/AD/cold,
Prepare, CPU pointwise/shaped and Checkpoint encoder explicitly rejected length mode. The later
implemented and accepted-target scopes are distinguished in Status above. Exact v5 field
encoding/compatibility tests are separately gated, not a task cascade. Dependency direction and
execution/resource owners do not change, so this documentation phase changes no architecture tests.

No fluent `.rnn`/fixed-scan migration, fixed BPTT, GRU/LSTM migration, higher-order waiver, implicit
model-wide mask/MEAN change, packing, new executor, low/Metal/vector/parallel/OpenBLAS route.
NN0032's advanced second-order CPU compile regression, singleton-CONCAT copy and unproved
mixed-engine compatibility remain separate gates. No performance/build run is part of this decision.

## Related documentation

The [focused explanation](../../architecture/general-differentiable-scan.md#accepted-length-aware-design-implementation-pending)
illustrates skipping and shared accumulation. [Checkpoint v5](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
owns future wire/bounds duties, not scan semantics. [ADR0045](0045-general-differentiable-scan.md)
and [ADR0047](0047-bounded-general-scan-graph-persistence.md) remain valid for the unchanged all-valid baseline.
