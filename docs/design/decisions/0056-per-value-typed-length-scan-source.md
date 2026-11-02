# ADR 0056: Per-value typed length-aware Scan source

## Status

Accepted; source Complete — 2026-11-01. The user approved the per-value Scan direction;
MAIN scoped source-only construction as the first milestone and accepted authority/Ready/GO
after independent pre-Java Class C APPROVE. Independent frozen source/tests/Javadoc review also
approves the implementation and consumer-rejection evidence. MAIN accepted Model0045 for source
only using retained source-matched coverage and three approved test-only recoveries; its
[Result](../../planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md#result)
retains original failed commands, with no clean root PASS claim. No downstream capability is added. The
[root](../../../ARCHITECTURE.md#scope-indexed-normative-contracts) indexes the sole
[Scan semantic owner](../../architecture/contracts/recurrent-scan.md#accepted-per-value-typed-length-aware-source).

## Context

Before Model0045, length-aware Model construction applied one homogeneous FLOAT32/FLOAT64 check
across all body declarations and values. Its row checker also rejected type-changing CAST. Ordinary
ScanBody containment already validates operation-local typing and requires each next carry to
match its own declaration. Thus source construction could not describe low stored carry
with explicit FLOAT32 work, or independent float/integer/BOOL slots, despite their otherwise
valid ordinary descriptors. Model source validity is not downstream capability.

Source inspection identifies independent guards in ScanBodyInference, ScanGradientRules,
PrepareContext and CpuGeneralScanLowering; SourceGraphReconstruction and GraphArtifactCodec reject
length mode outright. Direct negative regression tests preserve these boundaries, not relax them.
The former global dtype check implicitly enforced terminal SHARED contribution dtype; Model0045
replaces that implication with an explicit exact check alongside the contribution-shape check in Scan.

## Decision drivers

- Keep traversal/binding semantics separate from each ordinary operation's dtype and numerics.
- Permit useful typed source descriptions without implying execution or persistence support.
- Preserve row-role proof, state isolation, exact carry declarations and early failure.
- Keep existing fixed and ordinary unrolled APIs, kernels, constants and checkpoint behavior.

## Options considered

1. Keep the source-wide homogeneous restriction. This would preserve the old implementation but
   prevents valid per-value descriptions and makes source admission mirror CPU qualification.
2. Relax only Model length-aware source typing and CAST, retaining the closed operation inventory
   and independent downstream guards. This delivers one bounded construction capability.
3. Enable typed CPU/AD/cold/NN support together, or add a Scan-wide compute dtype. The former
   requires unselected execution and compatibility decisions; the latter assigns arithmetic
   policy to the loop instead of its ordinary operations. Neither is this task.

## Decision

Select option 2 for the accepted source-only implementation. The exact rules live only in the linked semantic owner:
all seven defined Model dtypes are admissible per value where existing ordinary operations in
the closed length-aware inventory permit them. Corresponding bindings remain exact; next-carry
shape/dtype is stable. Valid explicit CAST preserves row and dependency facts. A terminal SHARED
contribution must explicitly match its own carry dtype as well as shape. Scan adds no implicit
conversion, common working precision, new operation, type, signature or execution architecture.

## Rationale

For example, a FLOAT16 carry can be cast to FLOAT32 for ordinary ADD/MUL work and explicitly
cast back to FLOAT16 for the update. The body declares both represented sites. Scan only checks
that the returned carry descriptor matches the declaration and that row identity is preserved.
This avoids silently changing operation semantics or saved-state rounding. Allowing the same
description does not make existing F32/F64 consumers accept a low interior.

## Consequences

### Positive

Model construction can represent heterogeneous roles and explicit working-precision boundaries
without a second API or a graph-wide arithmetic setting. Unused discrete slots and identity
ROW carries no longer require artificial floating conversions.

### Negative and risks

More source graphs are valid but deliberately un-compilable or un-executable. Documentation
must distinguish those admissions. Global homogeneity previously hid the shared contribution
dtype invariant; omission of the explicit check would admit promoted updates. CAST must not
launder terminal row reductions or shared-carry dependencies into row feedback.

### Migration, testing, and follow-up

[Model0045](../../planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md) coordinates
the exact source/tests/Javadoc scope under the accepted review and MAIN GO. Direct raw-attachment,
graph and capability-query tests prove consumer rejection without relying on earlier guards.
The task's compact Result links the frozen source and validation artifacts, including retained
failed commands and targeted passes; it does not claim a clean full-module validation run.

All-valid Scan, ordinary unrolled `.rnn`/`.gru`/`.lstm`, static `Sequence.forward`, fixed kernels,
old checkpoints, constants and numerics remain intact. The eventual direction is NN recipes over
general Scan, with no Model RNN operation; removal or redirection is not selected now.
Higher-order support of unchanged unrolled APIs is not a blocker for this source step.
Generic CPU typed bindings/dtypes, feature SLICE, layout reads, cold/v5/legacy compatibility and
NN cutover remain coarse follow-up, not dependencies or newly authorized tasks/campaigns.

## Related documentation

- [Architecture explanation and conceptual typed example](../../architecture/general-differentiable-scan.md#per-value-typing-source-construction-only)
- [Foundational operation-local numerical rules](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)
- [Earlier length-aware decision](0049-length-aware-general-scan.md), retained for its original scope
