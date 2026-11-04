# CPU0057: Scan reverse body versus equivalent Java

## Reader context

This is the preserved CPU0057 comparison, not an audit of today's generated classes. Here
`CpuGeneralScanEmitter` builds the generated `invoke` method during preparation; later the
CPU prepared invocation calls that method with bound memory segments and numeric geometry.
The direct Java method E is a comparison oracle, not another production execution owner.
The [current Scan composition guide](../../../backend-guide/cpu-backend.md#current-general-scan-composition)
separates current admission from these legacy routes, while
[per-run resources and cold binding](../../../architecture/runtime-prepare-backend-boundary.md#per-run-resources-and-cold-binding)
explains why binding a segment is not copying its cells or transferring ownership of borrowed input storage.

For the count arithmetic below, let $B$ be the nonnegative integer batch size, $T=4$ the
number of time steps, and $A_t$ the active-row count at original time $t\in\{0,1,2,3\}$.
Then $S=\sum_{t=0}^{3}A_t$ counts active row-times and $TB-S$ counts inactive row-times.
At $B=7$, $S=5+3+2+1=11$ and $TB-S=28-11=17$. Thus the recorded zero-base
opportunity is $4S=44$ additions, while the inactive-cursor observation is
$20(TB-S)=340$ additions. These integer counts describe different bytecode sites;
neither equation estimates native instructions, elapsed time or floating-point rounding.

The absolute temporary-file evidence named below is unavailable to this reader revision.
Its recorded hashes, commands, outputs and conclusions remain historical evidence, not newly
reproduced results. In particular, the missing native-code and sample-to-version correlation
cannot be reconstructed from the summary tables.

## Purpose and conclusion

This diagnostic explains where the current generated Scan reverse differs from direct equivalent
Java, and what the retained evidence can actually say about that difference. It is an offline
report, not an [architecture contract](../../../glossary.md#architecture-contract), implementation approval, or new performance measurement.

The computation and checked memory-access counts match for the inspected [FLOAT32](../../../glossary.md#term-float32) case. The
[bytecode](../../../glossary.md#term-bytecode) does not match instruction for instruction. A small, proved redundancy remains: adding
a known-zero scratch base at two hot access sites. Omitting it would remove **44 additions at
B=7 and 244 at B=32**, plus the same number of local loads, in these invocation domains.
This is a code-shape opportunity only: [C2](../../../glossary.md#term-c2) may already fold it. It is not a demonstrated native
cost or a promised fix for an approximately 30% timing gap.

The retained just-in-time [compiler](../../../glossary.md#term-compiler) ([JIT](../../../glossary.md#term-just-in-time-compilation-jit)) [logs](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) show successful normal-entry C2 [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names) for
both implementations in every measured [JVM](../../../glossary.md#term-java-virtual-machine-jvm), with no inline failures in those normal tasks.
They contain neither native instruction listings nor [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-to-compilation-version timestamps.
Consequently, **the causal explanation of the timing difference remains unknown**. Code size,
temporary-local counts, and [C1](../../../glossary.md#term-c1) compilation failures do not supply that explanation.

## [Mental model](../../../glossary.md#term-mental-model) and comparison boundary

A Scan repeats a body over time. Its reverse consumes saved forward values rather than
recomputing [activations](../../../glossary.md#term-activation). Here the local activation [adjoint](../../../glossary.md#term-adjoint) is the [derivative](../../../glossary.md#term-derivative) contribution at
one active row/[feature](../../../glossary.md#term-feature); `dh`, `dX`, `dW`, `dU`, and `db` are cotangents of initial [carry](../../../glossary.md#term-carry), sequence,
input [weights](../../../glossary.md#term-weight), recurrent weights, and [bias](../../../glossary.md#term-bias) respectively.

```text
raw seed / scalar initialization + zero dX padding
    -> active time and row gates
    -> activation adjoint stored in one work bank
    -> ordered matrix contractions / bias reduction
    -> output carries and dX
```

G denotes the current generated `invoke`; E denotes the primary direct Java `reverseFloat32`.
E retains [runtime](../../../glossary.md#term-runtime) T, invocation B, actual initializer/derivative [scalar](../../../glossary.md#term-scalar) operands, and checked
unaligned access to match this current entry. Literal [oracle](../../../glossary.md#term-test-oracle) J is not the primary comparison.
This does not prove that T could never be baked under an appropriately qualified [specialization](../../../glossary.md#term-specialization)
and compatibility [identity](../../../glossary.md#term-identity-meanings). Likewise, current entry-pack loads are [ABI](../../../glossary.md#term-application-binary-interface-abi) observations, not
permanent architectural necessities.

The inspected case is [F32](../../../glossary.md#floating-point-precision-and-rounding), T=4, F=3, H=2, actual [Compiler](../../../glossary.md#term-compiler)-generated reverse, B=7 and B=32.
Lengths are `min(row % 5, 4)`. At original times 0,1,2,3 the active-row counts are respectively
`[5,3,2,1]` and `[25,18,12,6]`. Write A for one time's active rows, S for their sum
(11/61), and Q for active times (4/4). Dense row-times are 28/128; inactive row-times are 17/67.
These are domain-derived [bytecode](../../../glossary.md#term-bytecode) frequencies, not measured native instruction counts.

G orders consumers as activation, dh, dW, dU, db, dX; E uses activation, dX, dW, dU, db, dh.
All activation work is written before either implementation updates dh, and subsequent consumers
read the work bank, not old dh. This reorder does not change the inspected contractions' contributor
order or represented arithmetic. It is not a general authorization to reorder arbitrary bodies.

The owning constraints remain the [root invariants](../../../../ARCHITECTURE.md#core-invariants),
[CPU route contract](../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu))</span>,
[Scan arithmetic](../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
and [activity/raw-identity rules](../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories).

## [Identity](../../../glossary.md#term-identity-meanings) and reproducible evidence

Repository [admission](../../../glossary.md#term-admission) is HEAD/origin/main `07a281edd3086efde2853211b71b48288db027e1` plus the accepted
dirty baseline recorded by MAIN. CPU0056's rejected [binding](../../../glossary.md#term-binding) candidate was withdrawn; this report
inspects the accepted current bytes, not that candidate. No executable changes were made here.

All class names below are in `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`.

| [Target](../../../glossary.md#term-target) | Current retained identity | Code bytes / instruction sites |
|---|---|---|
| G | `Generated_9b77e0f7d675e786105a377bd85aa83f338c2a5f0532fdb3d1ef6162aa5093c5.invoke` | 2147 / 952 |
| E | `CpuSymbolicLengthScanEquivalentJavaOracle.reverseFloat32` | 1406 / 660 |

G's [SHA-256](../../../glossary.md#term-sha-256) is `30477241d7603cfec1c2f46079739a5ac10c2b1292dba3fb58aeaa2b05884330`;
E's is `79896bfa961e39abf2b3a18f5e40ff708cacd36d39de05bab41ed3024716521f`.
They match all three retained measured directories. G's [descriptor](../../../glossary.md#term-descriptor) is `(` + 19 copies of
`Ljava/lang/foreign/MemorySegment;` + `[JJJ)V`; E's is `(` + 18 copies of the same reference
descriptor + `JIZ)V`. The private E constructor is excluded from instruction counts.

The old `bbf6…` decompiler name is an [alias](../../../glossary.md#aliasing-and-arena), not the measured class identity. The pinned
`classfile-comparison.json` establishes full old/current byte equality after exactly one generated
[binary](../../../glossary.md#term-binary-descriptor-and-metadata)-name replacement; its old class [hash](../../../glossary.md#term-hash) is
`b119b85deb07068761d9755185b4d68758ddb90420b08eb7d0d7b19b1f79f02c`.
BCIs below refer to current pinned bytes, not that old name. A [bytecode index](../../../glossary.md#term-bytecode-index-bci) ([BCI](../../../glossary.md#just-in-time-compilation)) is an offset
inside one method; offset differences are not instruction counts.

Detailed evidence is retained off-tree in
`/private/tmp/synaptik-scan-body-audit.MvTjuv/`:

- [23-input SHA manifest](/private/tmp/synaptik-scan-body-audit.MvTjuv/inputs.json)<span data-glossary-navigation="true"> ([manifest glossary](../../../glossary.md#term-manifest))</span> and
  [manifest checker](/private/tmp/synaptik-scan-body-audit.MvTjuv/inputs-check.mjs).
- [Primary source audit](/private/tmp/synaptik-scan-body-audit.MvTjuv/source-audit.md),
  [current disassemblies](/private/tmp/synaptik-scan-body-audit.MvTjuv/source-classes.json),
  [static stage counts](/private/tmp/synaptik-scan-body-audit.MvTjuv/source-counts.json), and
  [domain-weighted counts](/private/tmp/synaptik-scan-body-audit.MvTjuv/source-hot-counts.json).
- [Primary JIT audit](/private/tmp/synaptik-scan-body-audit.MvTjuv/jit-audit.md)<span data-glossary-navigation="true"> ([JIT glossary](../../../glossary.md#term-just-in-time-compilation-jit))</span>,
  [structured JIT evidence](/private/tmp/synaptik-scan-body-audit.MvTjuv/jit-summary.json), and
  [reproducible log parser](/private/tmp/synaptik-scan-body-audit.MvTjuv/jit-parse.mjs)<span data-glossary-navigation="true"> ([log glossary](../../../glossary.md#term-log))</span>.

These absolute evidence links depend on retained local files; they are not portable repository
[artifacts](../../../glossary.md#term-artifact). The tables below preserve the principal findings if temporary files are later removed.
Original inputs are under `/private/tmp/synaptik-scan-binding.D2bvwa/`; the name-[normalization](../../../glossary.md#term-normalization)
record is under `/private/tmp/synaptik-scan-identity-verification.MFkNvn/analysis/`.

## Source intent to ClassFile: stage comparison

Source [owners](../../../glossary.md#term-owner) are [GeneralScan](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanEmitter.java),
Matmul (historical source path: `../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanMatmulEmitter.java`; no longer present in this checkout),
Reduction (historical source path: `../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanReductionEmitter.java`; no longer present in this checkout),
[Carrier](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java)<span data-glossary-navigation="true"> ([Carrier glossary](../../../glossary.md#term-storage-carrier))</span>,
and [E](../../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuSymbolicLengthScanEquivalentJavaOracle.java).
All source-line and [BCI](../../../glossary.md#term-bytecode-index-bci) ranges in this report are inclusive. Generated code has no invented
source-line mapping: emitter logic and operand [bindings](../../../glossary.md#term-binding) establish the correspondence.

The next table counts static instruction sites, including each stage's scaffolding. E column
ends with its source lines; the last column identifies generation source lines.

| Stage | G BCI; sites | E BCI; sites; source | Generator |
|---|---|---|---|
| Entry geometry/bindings | 0–122; 76 | 0–17; 10; 67 | GeneralScan 526–529, 862–920 |
| Raw dh [seed](../../../glossary.md#term-seed) copy | 123–186; 29 | 18–63; 19; 68–69 | GeneralScan 533–537, 1344–1370 |
| Actual dW initializer | 187–279; 44 | 106–155; 21; 72–73 | GeneralScan 1323–1332 |
| Actual dU initializer | 280–372; 44 | 156–205; 21; 74–75 | Same |
| Actual db initializer | 373–442; 32 | 206–255; 21; 76–77 | Same |
| Positive-zero dX | 443–488; 22 | 64–105; 20; 70–71 | GeneralScan 539–550 |
| Maximum/[scalar](../../../glossary.md#term-scalar)/time setup | 489–673; 97 | 256–381; 63; 78–87 | GeneralScan 552–620, 813–838 |
| [Activation](../../../glossary.md#term-activation) [adjoint](../../../glossary.md#term-adjoint) | 674–853; 84 | 382–543; 76; 89–100 | GeneralScan 1204–1248, 1408–1464 |
| dX | 1883–2133; 92 | 544–699; 76; 101–109 | Matmul 41–94 |
| dW | 1039–1244; 97 | 700–872; 83; 110–117 | Matmul 41–94; arithmetic tail |
| dU | 1245–1495; 97 | 873–1046; 83; 118–125 | Same |
| db [singleton](../../../glossary.md#term-singleton) decision | 1496–1504; 4 | 1047–1052; 3; 127 | Reduction 130–143 |
| db general fold | 1693–1882; 70 | 1121–1242; 56; 134–140 | Reduction 154–200 |
| db raw B=1 branch | 1505–1692; 69 | 1053–1120; 28; 128–132 | Same; not the B7/B32 path |
| dh | 854–1038; 88 | 1243–1395; 74; 142–150 | Matmul 41–94 |
| Outer increment/exit | 2134–2146; 7 | 1396–1405; 6; 151–152 | GeneralScan 653–662 |

### Equal semantics, arithmetic and memory traffic

Every consumed floating cell uses checked `MemorySegment.get(OfFloat,long):float` or
`set(OfFloat,long,float):void` with `JAVA_FLOAT_UNALIGNED`; lengths use
`get(OfLong,long):long` and `JAVA_LONG_UNALIGNED` ([Carrier](../../../glossary.md#term-storage-carrier) 284–301, 553–560).
The following per-stage operations are identical in G and E. Counts with A apply once per active time.

| Stage | Reads/writes and represented arithmetic |
|---|---|
| Initialization | 2B [seed](../../../glossary.md#term-seed) reads/dh writes; 3TB zero dX writes; 3 actual [scalar](../../../glossary.md#term-scalar) reads; 12 [parameter](../../../glossary.md#term-parameter) writes |
| [Activation](../../../glossary.md#term-activation) | 6A float reads; 2A work writes; 2A each seed ADD, saved square MUL, scalar SUB, [adjoint](../../../glossary.md#term-adjoint) MUL |
| dX | 6A work + 6A [weight](../../../glossary.md#term-weight) reads; 3A writes; 6A product/ADD pairs from +0 |
| dW | 6A x + 6A work reads; 6 old-[carry](../../../glossary.md#term-carry) reads/writes; 6A product/ADD pairs, then 6 distinct carry ADDs |
| dU | 4A history + 4A work reads; 4 old-carry reads/writes; 4A product/ADD pairs, then 4 carry ADDs |
| db, B>1 | 2A work reads; 2 old-carry reads/writes; 2A ADDs from +0, then 2 carry ADDs |
| dh | 4A work + 4A recurrent-weight reads; 2A writes; 4A product/ADD pairs from +0 |

Including the once-per-active-invocation [derivative](../../../glossary.md#term-derivative) scalar read, total float reads are
`2B+4+48S+12Q` = 594/3044; writes are `2B+3TB+12+7S+12Q` = 235/935.
Work-bank reads/writes are 242/22 and 1342/122. Length reads are `B+15BQ` = 427/1952:
B for maximum, then B each for activation/dX/dh, 6B dW, 4B dU, and 2B db per active time.
Extra G length reads, work-bank traffic, [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-by-parameter banks, activation replay and final
carry-copy passes are all zero in this pair.

Activity must dominate padded loads and dependent floating arithmetic. Inactive dh is raw
[identity](../../../glossary.md#term-identity-meanings); inactive dX is raw [positive zero](../../../glossary.md#nan-infinity-and-signed-zero). Saved activation outputs, actual scalar operands,
constituent [rounding](../../../glossary.md#term-rounding) sites, increasing contributor order, and distinct per-time parameter-carry
ADDs cannot be discarded. At B=1, [bias](../../../glossary.md#term-bias) SUM moves its sole contributor raw; at B>1, even A=1
does not replace the zero-start reduction with raw movement. [Signed zero](../../../glossary.md#term-signed-zero) makes that distinction observable.

### Geometry, address operations and loops

G has four numeric-pack `laload` sites, BCIs 6,12,18,26, reading T, 8B, B and 12B.
They execute once per invocation, not per row or contributor. E takes T/B directly and computes
two frame products at entry. Other recipe decisions are specialized away; generation-time maps,
callbacks and emitter objects are not [runtime](../../../glossary.md#term-runtime) allocations or [dispatch](../../../glossary.md#term-dispatch).

The table gives dynamic **[bytecode](../../../glossary.md#term-bytecode)** counts across all four times as G/E, in order
`long multiply; long add; long-local load; long-local store`. Local slots are not segment
memory accesses and are not hardware spills.

| Stage | B7: multiply; add; load; store | B32: multiply; add; load; store |
|---|---|---|
| [Activation](../../../glossary.md#term-activation) | 72/61; 88/99; 276/226; 116/44 | 372/311; 488/549; 1476/1226; 616/244 |
| dX | 160/358; 264/198; 485/61; 292/0 | 860/1958; 1464/1098; 2635/311; 1592/0 |
| dW | 264/480; 408/222; 900/282; 576/24 | 864/2280; 1608/1122; 3900/1182; 2376/24 |
| dU | 176/320; 272/148; 600/188; 384/16 | 576/1520; 1072/748; 2600/788; 1584/16 |
| db | 108/108; 44/22; 194/72; 108/8 | 508/508; 244/122; 894/272; 508/8 |
| dh | 116/248; 154/110; 298/28; 182/0 | 616/1348; 854/610; 1598/128; 982/0 |

G's advancing cursors save multiplications in every [MATMUL](../../../glossary.md#matrix-multiplication--matmul) stage but add local cursor traffic.
Neither side dominates merely by counting one [opcode](../../../glossary.md#term-opcode). Both have the same hot-stage loops and
conditional sites: activation 2/3, dX 3/4, dW 3/4, dU 3/4, db 2/3, dh 3/4
(loops/conditional sites). No additional B7/B32 body traversal is hidden here.

Initialization differs: G's dW/dU fills are nested, while E uses flat byte cursors. Across the
three [parameter](../../../glossary.md#term-parameter) fills, G/E have 22/0 address multiplies, 10/12 long additions, 25/15 loop-test
evaluations, and G adds three [constant](../../../glossary.md#term-constant) nonempty guards: +13 condition evaluations and +5 backedges.
Those are per-invocation differences, not per-time costs. [Scalar](../../../glossary.md#term-scalar)-read counts remain identical.
The raw dh copy also adds a known-zero base 2B times (14/64); dX zero-fill has equal cell counts.

G computes two time-frame products before its active-time gate; E computes them after it.
In these domains both perform eight such products, so the delta is zero. For all-zero lengths,
E returns after the maximum scan, whereas G still walks T with address/control setup while
skipping body loads and floating arithmetic. That separate case is not in the hot-count table.
E also retains a direction branch and `Math.max`; G bakes direction and emits a comparison.
Neither is hidden extra generated dispatch.

### Inactive [contraction](../../../glossary.md#contraction-dimension) rows: real [bytecode](../../../glossary.md#term-bytecode) work, unproved native cost

G dW's activity branch at 1118 jumps to 1159; dU's at 1342 jumps to 1391. Both destinations
advance two address cursors, including inactive rows, then increment the original row [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix).
The increments are 1159–1173 and 1391–1405. E's corresponding branches at 750 and 923 jump
to 811 and 985, skipping all operand-address expressions. Matmul source 73–87 makes the
distinction explicit: the gate encloses reads/accumulation, not cursor advances.

Six dW and four dU output cells each advance two cursors for every inactive row-time.
Thus G executes `20*(TB-S)` = **340/1340** inactive-path long additions, and equally many
cursor loads, stores and [stride](../../../glossary.md#term-stride) [constants](../../../glossary.md#term-constant); E executes zero of these operand-address operations
on that path. Dense length guards and row induction still execute in both. These counts are
not total G−E address deltas: the same cursor scheme saves active-path multiplications.

The following matched Java-equivalent excerpts are conceptual, not runnable APIs. `get` and
`length` abbreviate the same checked accesses; the [accumulator](../../../glossary.md#accumulator-type) starts at +0 for one dW cell.

```java
// G's address schedule; current sequenceFrame, k, col and batch are inputs.
long a = sequenceFrame + 4L * k, b = 4L * col;
for (int row = 0; row < batch; row++) {
    if (time < length(row)) contribution += get(x, a) * get(work, b);
    a += 12L;
    b += 8L;
}
// E's address schedule; same original-row domain and contributor order.
for (int row = 0; row < batch; row++)
    if (time < length(row))
        contribution += get(x, sequenceFrame + 12L * row + 4L * k)
                      * get(work, 8L * row + 4L * col);
```

For B=7, time=2, k=1, col=0, sequenceFrame=168. Rows 0–2 are inactive. G advances
`a: 172 -> 184 -> 196 -> 208` and `b: 0 -> 8 -> 16 -> 24`; E computes no operand address
until active row 3, then obtains the same 208/24. Neither reads [padding](../../../glossary.md#padding). Simply gating G's
existing increments would incorrectly compact coordinates. Changing to E's schedule is a
bounded alternative with an active/inactive tradeoff, not the preferred deletion below.

### Temporary locals and [compiler](../../../glossary.md#term-compiler)/decompiler noise

Extra G float-local stores, and separately extra loads, are [activation](../../../glossary.md#term-activation) `6S`, [MATMUL](../../../glossary.md#matrix-multiplication--matmul)/tails
`40S+80`, and db `24`: altogether `46S+104` = 610/2910 of each across hot stages.
Initialization adds `2B+3TB` = 98/448 of each. These are not extra work-bank reads/writes.

Activation's absolute store and load totals are different: G/E stores 154/88 and 854/488;
loads 198/132 and 1098/732. Both deltas are 66/366. This corrects the primary source report's
wording that loads always equal stores within a stage. Its extra-traffic totals remain correct.

Operand locals and producing-site locals often replace Java operand-stack [dataflow](../../../glossary.md#term-dataflow). Wide local
encodings enlarge G [bytecode](../../../glossary.md#term-bytecode) without adding arithmetic. Decompiled [aliases](../../../glossary.md#aliasing-and-arena) can obscure actual
assignments: G's `aload_1; astore 45` still exists at 40–41. Neither local names nor source-looking
statement counts measure [runtime](../../../glossary.md#term-runtime) work. No hardware spill follows from any of these observations.

## Existing [JIT](../../../glossary.md#term-just-in-time-compilation-jit) evidence, not inferred assembly

A compiled native method ([nmethod](../../../glossary.md#term-nmethod)) is an installed [JVM](../../../glossary.md#term-java-virtual-machine-jvm) code-[cache](../../../glossary.md#term-cache) object. [On-stack replacement](../../../glossary.md#term-on-stack-replacement-osr)
([OSR](../../../glossary.md#just-in-time-compilation)) enters compiled code at a loop [BCI](../../../glossary.md#term-bytecode-index-bci) rather than normal method entry. [Compile](../../../glossary.md#compile) IDs below
are scoped to one JVM; numeric helper-method IDs are scoped to one [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names) task.

The table contains all normal-entry [C2](../../../glossary.md#term-c2) level-4 [target](../../../glossary.md#term-target) versions. [Log](../../../glossary.md#term-log) root is
`/private/tmp/synaptik-scan-binding.D2bvwa/`; filenames are
`measured-N-hotspot_pidPID.log`. Last column gives task/install/done line numbers in that log.

| [Run](../../../glossary.md#term-run)/PID | Target/ID | Install uptime s | Total nmethod bytes | Instruction-section span | Task / install / done lines |
|---|---|---|---|---|---|
| 1/17211 | G/2648 | 0.866 | 22328 | 22032 | 33067 / 8938 / 46289 |
| 1/17211 | E/2649 | 0.849 | 18968 | 18672 | 345811 / 8936 / 358980 |
| 2/17219 | G/2582 | 0.745 | 22328 | 22032 | 146047 / 8611 / 159238 |
| 2/17219 | E/2605 | 0.888 | 18968 | 18672 | 79612 / 8742 / 92750 |
| 3/17226 | G/2612 | 0.896 | 22328 | 22032 | 84072 / 8742 / 97263 |
| 3/17226 | E/2585 | 0.648 | 18984 | 18688 | 136170 / 8628 / 149308 |

G hidden suffixes are respectively `/0x000001e0010d8c00`, `/0x00001ff8010d8c00`,
`/0x00001f00010d8c00`. The exact class/method [descriptors](../../../glossary.md#term-descriptor) and [hashes](../../../glossary.md#term-hash) correlate these to the
retained bytes; S/N harness routes share the same raw generated [MethodHandle](../../../glossary.md#native-numerical-library-and-instruction-architecture), not separate classes.

All 34 target tasks were checked: 28 successful C2 (6 normal, 22 OSR) and 6 failed [C1](../../../glossary.md#term-c1) level-3
OSR tasks, with no successful target C1 nmethod. Per-run task counts are 12/12/10. Each normal
C2 task has 1140 logged inline successes and zero failures. Each has 27
`AbstractMemorySegmentImpl.get` and 11 `set` successes; segment/[layout](../../../glossary.md#layout)/bounds/session checks
each have 38. These are log decision occurrences, not executed calls or proof of eliminated checks.

The C1 failures say `out of virtual registers in LIR generator`; they are [compiler](../../../glossary.md#term-compiler) resource
failures, not observed C2 hardware spilling. Each G C1 task has 1210/348 inline successes/failures;
each E C1 task 1183/340. The only inline failure in successful C2 tasks is one E OSR task per
JVM: IDs 2640/2596/2582, OSR BCIs 1136/1136/897, failure log lines 102780/77600/134183,
reason `member_name not constant` in the long-valued foreign-memory path. None is a normal-C2 failure.

For exact target-root methods/compile IDs the retained [runtime](../../../glossary.md#term-runtime) tty records contain zero
`uncommon_trap`, `deoptimized`, or `make_not_entrant` events. Two tier-3 OSR
`make_not_compilable` events per JVM do not prohibit the later C2 compilations.
Compile-time trap records are different: for example, run-1 G task 2648 has four root pack-array
range-check trap records, and both normal targets have 38 inlined `checkBounds` records.
They do not count traps firing or surviving checks executed per [element](../../../glossary.md#term-element). Missing log events
are bounded negative observations, not proof of universal absence.

The instruction-section span is literally `stub_offset - insts_offset`, possibly including
[padding](../../../glossary.md#padding); `insts_offset=248`, `size-stub_offset=48`, and `task_done.nmsize=span+8` here.
G's normal total size exceeds E by 3360/3360/3344 bytes. None of these fields identifies
executed instruction count, register allocation, bounds-check elimination, or floating
multiply-add ([FMA](../../../glossary.md#term-fused-multiply-add-fma)) [fusion](../../../glossary.md#term-fusion). [Bytecode](../../../glossary.md#term-bytecode) product/ADD sites do not by themselves establish native fusion.

The retained `Cpu0056ScanBinding.java` lines 102–117 keep phase/round/target/calls/elapsed nanos,
but discard the [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) start and record no JVM uptime brackets. Hence normal C2 installation
at 0.648–0.896 s cannot be placed before the measured window or a particular B/phase/sample.
Availability during the JVM run is proved; measured-window C2 readiness is not.

## One preferred bounded next step

Subject to separate implementation authorization, propagate the existing zero-base proof to the
generation-proved zero-offset, nonbanked scratch cut. GeneralScan 554–558 binds its base;
`load` 1427 and `store` 1462 already have proof-aware omission paths. In current [bytecode](../../../glossary.md#term-bytecode),
local 108 is assigned zero once at 496–497 and never reassigned. [Activation](../../../glossary.md#term-activation) loads it at 823
and adds it at 825; db does so at 1785–1788. Each site executes 2S times, totaling 4S=44/244
zero additions and matching local loads. E has no such storage-base addition.

Conceptual matched extraction, not an implemented patch:

```java
// Current G; workBase is proved zero for this cut's entire lifetime.
work.set(JAVA_FLOAT_UNALIGNED, relative + workBase, adjoint);
// E / proposed bounded shape: same checked access and represented value.
work.set(JAVA_FLOAT_UNALIGNED, relative, adjoint);
```

Concrete interpretation: B=7, time=2, active row=3, col=1 gives `relative=8*3+4=28`.
With actual [scalar](../../../glossary.md#term-scalar) operand 1, dh [seed](../../../glossary.md#term-seed) 0.25, emission seed 0.5 and saved output 0.5,
the represented steps are seed=0.75, square=0.25, slope=0.75, [adjoint](../../../glossary.md#term-adjoint)=0.5625.
Both stores write 0.5625 at byte 28; db later reads the same represented cell. An inactive
row must still skip both sites. The scalar value 1 is this example's input, not a proposed literal substitution.

Require zero coefficients for every admitted B, exact [lifetime](../../../glossary.md#term-lifetime)/[alias](../../../glossary.md#aliasing-and-arena) proof, no changing bank,
unchanged activity/saved-value/[rounding](../../../glossary.md#term-rounding) behavior, and unchanged cell-relative [cache](../../../glossary.md#term-cache) [scope](../../../glossary.md#term-scope).
Do not infer zero from one sampled B or apply it to time-dependent sequence/history bases.
Any future change also needs separately authorized [identity](../../../glossary.md#term-identity-meanings) and focused-[validation](../../../glossary.md#term-validation) handling.
This recommendation does not claim removal of every address temporary or [MATMUL](../../../glossary.md#matrix-multiplication--matmul) base addition.

The inactive-cursor schedule is an additional observation, not a second implementation request.
Before attributing the timing gap to either shape, the missing evidence is the exact relevant
normal G/E native instruction/[dataflow](../../../glossary.md#term-dataflow), its executed-path relevance, and [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-to-version/uptime
correlation. Existing [logs](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) cannot supply those facts. No new [run](../../../glossary.md#term-run) or benchmark campaign is approved
or requested by this report.

## Independent review, [validation](../../../glossary.md#term-validation) and limits

The clean reviewer inspected the pinned current disassembly, named sources and focused E tests,
checked [BCI](../../../glossary.md#term-bytecode-index-bci) branches and the sole zero-base assignment, recomputed domain/count formulas, and
cross-checked raw [JIT](../../../glossary.md#term-just-in-time-compilation-jit) task blocks/install records rather than relying only on report summaries.
The [activation](../../../glossary.md#term-activation) load/store wording correction above is incorporated; no open P1/P2 remains.

Offline validation commands and results:

```sh
node /private/tmp/synaptik-scan-body-audit.MvTjuv/review-source-check.mjs
node /private/tmp/synaptik-scan-body-audit.MvTjuv/jit-parse.mjs --check-summary
node /private/tmp/synaptik-scan-body-audit.MvTjuv/review-jit-check.mjs
node /private/tmp/synaptik-scan-body-audit.MvTjuv/inputs-check.mjs check
node /private/tmp/synaptik-scan-body-audit.MvTjuv/review-report-check.mjs
git diff --check -- docs/planning/backends/cpu/0057-scan-body-java-audit.md
```

Source checks reproduce both pure-[Node](../../../glossary.md#node) count outputs and verify instruction/stage [partition](../../../glossary.md#partition),
exact proof sites, active domains and count deltas. JIT checks reproduce the primary summary
and independently verify all 34 raw [target](../../../glossary.md#term-target) tasks, [nmethod](../../../glossary.md#term-nmethod) fields and target [runtime](../../../glossary.md#term-runtime)-event [scope](../../../glossary.md#term-scope).
The [manifest](../../../glossary.md#term-manifest) [checkpoint](../../../glossary.md#term-checkpoint) covers 23 unchanged inputs; the additional [run](../../../glossary.md#term-run)-2/3 class pins are
separately verified. Report checks validate local links/anchors, arithmetic examples and whitespace,
including this untracked Markdown file. No Java workload, [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names), test, witness, benchmark,
new JIT or Git mutation was performed by this review.

Prior accepted semantic validation is reused, not rerun. The focused tests
`cpu0045EquivalentJavaActualReverseRuntimeOperandsAndEmptyDomains` and
`cpu0045EquivalentJavaRawSingletonSpecialsAndInactiveBodyReads` were inspected, not executed;
they cover actual operands, empty domains, [padding](../../../glossary.md#padding), raw [singleton](../../../glossary.md#term-singleton) behavior and inactive expired
inputs. This report does not upgrade their historical timing evidence or claim new executable validation.

Documentation profiles: general/[planning](../../../glossary.md#term-planning), with developer-guide/example conventions for the
explanatory snippets. Targeted glossary review found no new domain term. Only this diagnostic
report is changed in the repository; public [API](../../../glossary.md#term-application-programming-interface-api), behavior, architecture and [Javadoc](../../../glossary.md#term-javadoc) are unchanged.
MAIN retains task acceptance and shared-plan [synchronization](../../../glossary.md#term-synchronization). All native-cost and phase-correlation
limits above remain explicit research questions, not incomplete work within this offline scope.

Independent review: **APPROVE; no open P1/P2.**

Status: Complete
