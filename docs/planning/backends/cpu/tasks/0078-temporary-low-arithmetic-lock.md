# Task 0078: Temporary Java CPU low-arithmetic lock

## Status

Complete — independent final production/evidence/documentation APPROVE0 and MAIN's full affected
gate/public consumers passed. The low implementation remains retained, not expanded; no final
production-repository or whole-release qualification is claimed.

## Change class

Class C — capability, direct generated realization/cache reuse and fixed-recurrent admission.
Clean implementation, independent code/Javadoc/contract review and actual consumer validation.

## Goal

Temporarily reject BFLOAT16/FLOAT16 floating computation on the single CPU backend while keeping
shared Model types, low host/storage/borrow/export, raw movement, initialized literals, all existing
casts/conversions, INT32/INT64/BOOL control and exact represented-value semantics. No deletion,
new low algorithm, runtime unlock flag, hidden F32 fallback or precision/profile object.

[The reviewed matrix](../../../../backend-guide/production-support.md#temporary-java-low-computation-lock)
sets the operation-sensitive boundary. Reject low numerical operand roles even with a promoted
result: ADD/SUB/MUL/DIV/POW and numerical unary composites, MATMUL/convolution, SUM/MEAN/PROD,
arithmetic/statistical/masked/cumulative reductions, average pooling, normalization, losses,
attention, dropout scaling, floating scatter ADD/MUL, and CPU's zero-initialized fold addition.
CPU FoldIr always declares addition from positive zero: do not invent a no-overlap/raw alternative
or singleton exception. Fixed low RNN/GRU/LSTM compute is locked even for T/B=0 or all-zero lengths.

Preserve exact ABS/NEG/SIGN/FLOOR/CEIL/RELU/clamp, MIN/MAX including reductions/scatter, stored
comparisons/classification/WHERE, ordering/arg-extrema/max pooling, raw affine/movement/indexing/
replacement, explicit casts and raw state/mask transitions under their existing predicates. Do not
ban a partition merely because it contains low storage. State/mask control is not arithmetic;
INITIAL_STATE stays available, while low dropout's rescaling is locked without p=0 shortcuts.

Explicit low parameter/storage -> F32 CAST -> F32 computation remains valid, with the ordinary
last gradient/storage conversions and existing Training-owned F32-working optimizer updates.
The backend lock must not disable generic Training storage conversion/update algorithms.

## Contracts and prerequisite coordination

- [Root invariants](../../../../../ARCHITECTURE.md#core-invariants): one Model semantics; current
  low guarantees are implementation-retention facts, not an exception to the accepted CPU lock.
- [CPU0078 exact authority](../../../../architecture/contracts/backend-execution.md#cpu0078-temporary-low-arithmetic-lock),
  [low evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)
  and [identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity).
  Root/owning CPU-only lock and exact cold signatures are accepted; Metal and Model semantics stay.
- [Prepare/runtime owners](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  retain slots, resource ownership and hot execution. No Runtime low-policy check or new protocol.
- [Product authorization](../../../production-release-goal.md#java-cpu-target) requires the lock;
  independent review resolves source/seam concerns without a duplicate operation registry.

## Concrete ownership and proposed cold seams

No new/moved production type, package, public facade, dependency or opcode ordinal. Existing
CpuPointwiseOpcode owns one numerical-versus-exact opcode distinction, reused through the existing
operation-to-opcode mapping. Existing internal.lowering.CpuPartitionLowering owns the CPU-local
availability checks, not a new policy/service/registry. Accepted exact cold-only signatures:

- `public static boolean permitsLowPrecisionUse(OperationCapabilityQuery query)`;
- `public static void requirePermittedLowPrecisionUse(CpuPortableKernelIr ir)`;
- `public static void requirePermittedLowPrecisionUse(CpuKernelSpecialization specialization, CpuKernelIr ir)`.

These methods are unsupported backend-internal API despite cross-package public visibility.
The query predicate never calls provider.supports, recurrent/general-Scan analysis or the
CpuPartitionLowering constructor. Generalize the one existing `opcode(CompiledNode)` mapping to
`opcode(OperationKind)` and migrate Lowering, GeneralScanLowering and DagDecomposer callers;
do not copy a provider mapping. POW remains numerical even under IDENTITY/POSITIVE_ONE realization.
The opcode property is `usesFloatingArithmetic()`. Authorized narrow getters are existing
CpuAggregateIr/CpuPool2dIr/CpuPool3dIr.encodedKind(CpuKernelIr), CpuScatterIr.encodedReduction
and CpuRandomIr.encodedFamily, returning only their existing semantic enums; see exact authority.

Use the existing exact typed family metadata/encoding owners. Do not copy shape inference,
normalization/convolution/fold mathematics, parse a second ad-hoc family registry or convert a
query into a fake graph. Pointwise checks inspect actual numerical inputs and virtual values,
not every low boundary indiscriminately; explicit CAST boundaries stay meaningful. The shared
policy is compile-time backend admission, not a selectable production mode.
Only narrow semantic-discriminator access on the existing aggregate/pool/scatter/random encoding
owners is permitted, shared with their existing emitter decoder where needed. No general
decoder, geometry reconstruction or decode-record allocation just for availability. Authenticate
the facts emission actually consumes: typed MATMUL encoding/key must agree with its envelope,
and both actual represented operands are checked, not only the promoted output.

Migrate `CpuClassFileKernelGenerator.defineClassBytes` to `(specialization, kernelIr, classBytes)`
so direct byte definition can authenticate the current exact IR before definition, just like fresh
generation. Migrate every caller/test; retain no two-argument bypass or compatibility shim. Existing
verify-only class-shape inspection need not execute/admit an otherwise locked algorithm.

## Independent admission boundaries

1. CpuCapabilityProvider and direct ordinary/family/fixed-recurrent lowering reject locked source
   occurrences, before absorbed-one/count/fusion paths erase source roles.
2. Direct preparer/retained lowered plans authenticate typed facts independently before resources,
   candidates or generated realization; all CPU routes obey the lock, with no native escape.
   Sweep analyze/analyzeSelected/completePlanCandidates before declarations/candidates. In
   finalization, sweep retained semantic units AND actual represented portable plans before
   realizing copies/children or configuring native resources; representationUnits can substitute plans.
3. ArtifactStore authenticates current IR/policy before weak-intern or trusted-root hits; generator
   fresh emission and direct byte definition independently reject. Never silently retry old low code.
   The low CpuConstantWriteIr / constant-write-v1 physical family is explicitly rejected too:
   it represents arithmetic SUM-derived counts whose source roles have been erased, not original
   graph-literal initialization. Reject it at typed/envelope/retained-finalization admission.
   Shared Prepare's original low graph-literal initialization remains available. Cache policy
   rejection stays outside persisted catches and must not turn into miss/regeneration.
4. Fixed recurrent lowering/preparer/executable binds only current F32/F64 computation; retained
   concrete low kernels are not removed or reimplemented. Raw primitive/storage helpers are not
   themselves an executable backend admission API.
Direct MATMUL checks both lowering overloads at lowerRecognized, including recognized/fused
decomposer paths that previously lacked anchor-provider admission. Fixed executable construction
checks before zero-scratch decisions or concrete low-kernel construction. Existing OpenBLAS
routes are already F32/F64-only and remain so; no native algorithm change.

Keep the policy outside hot loops and avoid unnecessary cold object allocation/copying. Inspect
internal low values and specialized instruction-free envelopes, mixed/promoted MATMUL operand
roles, low-only source folds, and T/B=0/all-inactive cases. Early no-work must not bypass admission.
Unchanged retained/allowed emitted code requires no arbitrary global generator/ABI bump; any real
changed bytes/facts need their owning selective identity decision before code.

## Dependencies and integration

- Depends on: reviewed scope freeze, complete CPU0077 integration, reviewed seams and authority.
- Conflicts with: CPU provider/lowering/preparer/IR/generator/cache/fixed execution, current low
  ledger and affected tests. Serialize the entire CPU source cutover; no all-PURE Scan writer.
- Parallel group: RELEASE-CPU78-LIFECYCLE; sole CPU writer, disjoint Engine0022 cleanup scope.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted post-CPU0077
  cpu0078-base.json and MAIN's authority-only Engine0022 addendum. No CPU source/semantics change
  accompanies that clarification; MAIN synchronizes its named documents into the isolated worktree.
- Integration order: scoped approval/authority -> isolated implementation -> independent review ->
  MAIN gates/public low-storage/negative compute smoke -> later all-PURE composition/final RC join.
- Integration validation: final CPU/Javadoc gate, low-ledger/current conformance, TrainingSessionCpu,
  plain-JVM/MLP/CNN/RNN and new low-negative/raw controls; MAIN groups focused selectors once source
  and independent review stabilize, then unchanged LowLockProductionSmoke after and later RC join.
- Shared-document integration owner: MAIN; worker owns only declared CPU/test/ledger source.

## Acceptance and test changes

Both low dtypes reject arithmetic across source/provider, direct typed generation/definition,
pre-cache reuse and fixed execution, including promoted outputs and empty/inactive domains. Valid
F32/F64 arithmetic and low-storage explicit F32 computation remain exercised. Check exact raw bits,
signed zero/subnormals, winner/tie/index/mask/state results and all existing conversions.

Update current ledger v2 from actual provider answers, never historical v1 or Metal rows. Keep an
honest coverage inventory: low numerical rows become explicit unsupported outcomes, not skipped,
deleted from the domain or falsely GENERATED. Remove incidental exact-count/name/wording pins;
do not re-pin new totals. Permanent tests target consumer-visible unsupported behavior, storage
preservation, current admission and stale-artifact rejection, not mock echoes/source-text checks.

Existing softmax validators/reference/emitters keep their retained low implementation; newly
unsupported generation/consumer cases must become meaningful capability negatives. Do not expand
or numerically repair the frozen low implementation. Tests run once by MAIN after stable review;
worker skips build/lint/tests/formatters. Javadoc reflects current admission and retained semantics.

Independent pre-Ready review: one P2 count-write ambiguity identified; the explicit physical-family
rejection above resolves lost source provenance without a new type/fact or blocking original
literals. Opcode mapping, exact encoded-discriminator reuse, typed MATMUL consistency and actual
representation-finalization boundaries are part of this packet, not follow-up work.

MAIN Ready verification: complete predecessors, accepted reviewed scope, sole frontier/inactive
conflicts and exact root/CPU authority verified before launch. ADR0060 records the Class C decision;
no new production type/dependency, emitter mathematics, scheduler/native/ABI or schema change.
Before evidence: actual public low ADD and explicit F32/cast-back/exact raw ABS for both types,
plus INT64 arithmetic. The BF16 successful prefix and corrected separate FP16 run are retained;
original wrong-FP16-import command stays FAILED, not relabeled PASS.

Genuine old ADD artifacts (BF16 class projection59, F16 projection68) were generated/defined by
the unmodified old generator in low-lock-authentic-legacy, with class/identity/compatibility/key
payloads and provenance.json. Use these for stale-artifact fixtures without a production bypass;
retain honest failed-first/corrected-F16 capture evidence. Do not regenerate them after the lock.

## Result

Complete after independent final production, evidence and documentation acceptance. No
full-production repository checkpoint or release-readiness claim follows.

The 23 production files use the existing opcode mapper and CpuPartitionLowering availability
owner. Provider/source, portable and encoded retained IR, all preparation paths, fresh generation,
direct definition, persisted/weak cache reuse and direct fixed recurrence reject low numerical
operand roles before execution/resources. Direct definition requires spec+actual IR+bytes; no
two-argument bypass, unlock flag, new production type/profile or duplicated arithmetic remains.
Exact/discrete/raw/conversion/storage behavior and the original literal initialization are retained.
The ordinary symbolic MATMUL typed/envelope seam authenticates its actual existing encoding;
qualified generated keys/classes are preserved, not globally versioned.

MAIN checked every child SHA and frozen MAIN preimage before exact integration. Independent
Cpu0078DesignReview approved production and its MATMUL correction; Cpu78BehaviorReview and
Cpu78CatalogReview approved final disjoint fixture repairs and resource/evidence changes.
Obsolete count/wording pins were deleted, not repinned. The copy-only Gap JUnit test was replaced
by the non-JUnit CpuSpecializedSemanticClosureInventory CLI; actual semantic-execution owners
and the complete canonical owner universe remain. Its seven-column derived resource records
EXACT_ROW_OUTCOME_OWNERS, not an execution certificate.

Actual current rendering retains all 41,601 original owner IDs/forms. All 8,700 still-generated
rows are literal-equal in all 27 fields, including IR/entry/key/class bytes. Original low-typed
numerical occurrences account for 32,636 generated-to-provider-rejected and eight former
preparer-to-provider-rejected transitions; denied rows have actual false admission and no
fabricated generated facts. Current v2 retains 290 original queries with 54 CPU-only support
losses; all 145 Metal rows and historical v1 remain literal. The real non-JUnit CLI preserves
all 29 derived forms/raw counts/owner methods and independently checked current group counts.

MAIN's final affected command passed (artifact549/bg99, 177.03s):
`:backends:cpu:test/javadoc/jar`, `:testing:backend-conformance:test`,
`:testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest'`,
`:modules:engine:test/javadoc`. CPU: 1,471 passed/60 skipped; conformance: 83 passed; Training CPU:
eight passed/two Metal-configured skips; Engine: 154 passed/two skips; zero failures/errors.
The 60 CPU skip identities are unchanged from the first full gate; retain their complete
performance/diagnostic/evidence list, not an assertion that all are timing tests. The final
command executed two tasks and reused 51 unchanged task results, including unaffected component
and Javadoc/jar evidence. CPU test execution was fresh after the reviewed repairs.

Unchanged actual public before/after low consumers now reject BF16/F16 ADD while explicit
F32 computation/cast-back, raw ABS signed-zero/subnormal controls and INT64 arithmetic pass.
A separate real TrainingSession consumer passes both low storage types with six changing-target
steps, exact F32 master bits, logical narrowing, prepared-graph reuse, snapshot restoration and
fresh resumed steps; caller-owned Arena lifetime remains intact. This is retained F32 work,
not low numerical backend expansion.

Evidence lives in MAIN's source-snapshot run directory: cpu0078-final-gate-evidence.json and
copied XML; cpu0078-public-after-evidence.json; cpu0078-low-master-evidence.json;
cpu0078-actual-resource-authentication.json; cpu0078-specialized-inventory-authentication.json;
exact cumulative/integration manifests. Failed attempts 450/462/480/487/496/502 stay FAILED:
compile/fixture repairs, first snapshot comparisons and the wrong-task selection were not
relabelled as passes. Genuine old BF16/F16 ADD class/identity/compatibility/key payloads and
their capture provenance remain, never regenerated through a current production bypass.
