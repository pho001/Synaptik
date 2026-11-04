# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0060: Temporary [CPU](../../glossary.md#term-cpu) low-arithmetic [admission](../../glossary.md#term-admission) lock


## Editorial reading context

CPU admission asks whether a numerical role consumes low-precision values, not merely whether the final output dtype is low. Explicit conversion into FLOAT32 work is a different operation from secretly emulating unsupported low arithmetic. This temporary-lock decision is historical policy; consult the current support table rather than treating its old exclusions as today's provider answers.

Use the [current qualification and lock reconciliation](../../backend-guide/production-support.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted design under the user's first-production authorization, after independent [scope](../../glossary.md#term-scope) and
source/seam review. Root and the owning [backend](../../glossary.md#backend) contract [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) the [CPU](../../glossary.md#term-cpu)-only cutover before
CPU0078 [Ready](../../glossary.md#term-ready). Implementation and numerical/consumer acceptance are recorded by
[CPU0078](../../planning/backends/cpu/tasks/0078-temporary-low-arithmetic-lock.md#result), not inferred
from this decision. The [reviewed matrix](../../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([matrix glossary](../../glossary.md#term-matrix-meanings))</span> remains the
[release](../../glossary.md#term-release) [target](../../glossary.md#term-target)/evidence ledger.

## Context

The user explicitly chose to retain, not expand and temporarily lock Java [CPU](../../glossary.md#term-cpu) [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16)
computation, while requiring [Metal](../../glossary.md#term-metal) production support. Storage [dtype](../../glossary.md#data-type--datatype) alone cannot decide arithmetic
permission: exact selection, comparisons, raw movement and explicit conversion remain useful to
[F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) networks, and an output can be promoted while still consuming a low numerical operand.
Current prepared/[cache](../../glossary.md#term-cache)/fixed-recurrent entry points cannot rely only on a fresh provider query.

## Decision

Keep the implementation and [Model](../../glossary.md#term-model) numerical semantics. Introduce no profile, [runtime](../../glossary.md#term-runtime) opt-in,
policy object, hidden [F32](../../glossary.md#floating-point-precision-and-rounding) fallback, alternate [backend](../../glossary.md#backend) or new production type/package. Existing
[opcode](../../glossary.md#term-opcode)/[lowering](../../glossary.md#lowering)/[IR](../../glossary.md#term-intermediate-representation-ir) [owners](../../glossary.md#term-owner) hold one [CPU](../../glossary.md#term-cpu)-local availability distinction, outside execution loops.
The [owning CPU contract](../../architecture/contracts/backend-execution.md#cpu0078-temporary-low-arithmetic-lock)
defines exact preserved/numerical families and the bounded cold [signatures](../../glossary.md#term-signature).

Authenticate at source/direct lowering, preparation and retained finalization, before any [cache
hit](../../glossary.md#cache-and-memoization) or class definition, and independently before direct fixed-recurrent construction. Check actual
represented realization plans as well as semantic units. Numeric virtual values and typed [MATMUL](../../glossary.md#matrix-multiplication--matmul)
operands matter; a low boundary feeding an explicit F32 CAST does not prohibit later F32 arithmetic.

SUM-derived [constant](../../glossary.md#term-constant)-write IR has lost its numerical [provenance](../../glossary.md#provenance). Reject that low physical family
at retained/IR [admission](../../glossary.md#term-admission); original [graph](../../glossary.md#graph)-literal initialization remains shared [Prepare](../../glossary.md#prepare) work and
stays available. Do not add provenance solely to rescue this temporary optimization.

Direct generated-byte definition receives the exact [canonical](../../glossary.md#term-canonical-form) IR in addition to [specialization](../../glossary.md#term-specialization)
and bytes. Migrate every internal caller, with no two-[argument](../../glossary.md#term-argument) bypass. Policy rejection precedes
weak/persisted [cache](../../glossary.md#term-cache) reuse and stays outside miss/regeneration catches. Verify-only class inspection
retains its separate non-executing purpose and current trusted-local-byte threat model.

## Preservation and compatibility

All shared types, storage/[borrowing](../../glossary.md#term-borrowing)/export, original literals, raw exact movement, conversions,
[BOOL](../../glossary.md#data-type--datatype)/integral control and exact [unary](../../glossary.md#term-unary)/predicate/[extrema](../../glossary.md#saturation-pole-and-extremum)/ordering/winner semantics remain. Explicit
low-storage -> [F32](../../glossary.md#floating-point-precision-and-rounding) compute -> final storage conversion stays valid, including Training-owned
F32-working master/candidate updates. This does not authorize implicit emulation of a low [operation](../../glossary.md#operation).

No emitter mathematics, low [kernel](../../glossary.md#kernel) or generic [optimizer](../../glossary.md#term-optimizer) is deleted or repaired by this lock.
[OpenBLAS](../../glossary.md#term-openblas) already restricts computation to F32/[F64](../../glossary.md#floating-point-precision-and-rounding); the integral-only partial-reduction [IR](../../glossary.md#term-intermediate-representation-ir) retains
its existing exclusion of every floating [dtype](../../glossary.md#data-type--datatype). Empty/inactive work cannot bypass [admission](../../glossary.md#term-admission).

Allowed generated bytes, entry [ABI](../../glossary.md#term-application-binary-interface-abi) and [identities](../../glossary.md#term-identity-meanings) remain unchanged. The definition method's
extra cold [argument](../../glossary.md#term-argument) is internal source migration, not a generated/native ABI. No arbitrary global
[schema](../../glossary.md#term-schema) bump is warranted; any demonstrated changed emitted fact requires its own selective review.
Current ledger v2 records actual new [CPU](../../glossary.md#term-cpu) answers; historical v1 and every [Metal](../../glossary.md#term-metal) answer stay intact.
Unlocking needs an explicit later decision and requalification, not a property or retry.

## Verification boundary

Before the cutover, actual public [BF16](../../glossary.md#floating-point-precision-and-rounding) and [FP16](../../glossary.md#term-fp16) ADD and explicit [F32](../../glossary.md#floating-point-precision-and-rounding)/cast-back/exact-ABS controls
were observed. Authentic old ADD Class-Files were generated and defined by the old generator for
both [dtypes](../../glossary.md#data-type--datatype); these support real stale-[artifact](../../glossary.md#term-artifact) rejection fixtures without a production bypass.
The first [fixture](../../glossary.md#term-fixture) commands' [F16](../../glossary.md#floating-point-precision-and-rounding) import/projection mistakes remain failed evidence; corrected F16
parts were exercised separately rather than relabeling the commands PASS.

Acceptance must prove unsupported numerical roles at all entry points, old-[cache](../../glossary.md#term-cache) rejection,
retained [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity)/[signed-zero](../../glossary.md#nan-infinity-and-signed-zero)/[subnormal](../../glossary.md#term-subnormal)/discrete results, storage/F32 workflows and unaffected
F32/[F64](../../glossary.md#floating-point-precision-and-rounding) neural journeys. No source, [operation](../../glossary.md#operation) count or ordinary [JVM](../../glossary.md#term-java-virtual-machine-jvm)-only success is by itself
first-[release](../../glossary.md#term-release) or [Metal](../../glossary.md#term-metal) production readiness.
