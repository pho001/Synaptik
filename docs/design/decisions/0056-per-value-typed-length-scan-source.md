# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0056: Per-value typed length-aware [Scan](../../glossary.md#general-differentiable-scan) source


## Editorial reading context

A body can describe FLOAT16 stored carry, explicit FLOAT32 working operations and an explicit cast back. Scan checks each value's own descriptor; it does not choose a global compute type. Source validity alone does not make Compiler AD, CPU execution or persistence accept that body. The source-only status and retained downstream rejection tests are essential limits.

Use the [typed source example and consumer boundaries](../../architecture/general-differentiable-scan.md#per-value-typing-source-construction-only) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted; source Complete — 2026-11-01. The user approved the per-value [Scan](../../glossary.md#general-differentiable-scan) direction;
MAIN scoped source-only construction as the first milestone and accepted authority/[Ready](../../glossary.md#term-ready)/GO
after independent pre-Java Class C APPROVE. Independent frozen source/tests/[Javadoc](../../glossary.md#term-javadoc) review also
approves the implementation and consumer-rejection evidence. MAIN accepted Model0045 for source
only using retained source-matched coverage and three approved test-only recoveries; its
[Result](../../planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md#result)
retains original failed commands, with no clean root PASS claim. No downstream capability is added. The
[root](../../../ARCHITECTURE.md#scope-indexed-normative-contracts) indexes the sole
[Scan semantic owner](../../architecture/contracts/recurrent-scan.md#accepted-per-value-typed-length-aware-source)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>.

## Context

Before Model0045, length-aware [Model](../../glossary.md#term-model) construction applied one [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) check
across all body declarations and values. Its row checker also rejected type-changing CAST. Ordinary
ScanBody containment already validates [operation](../../glossary.md#operation)-local typing and requires each next [carry](../../glossary.md#term-carry) to
match its own declaration. Thus source construction could not describe low stored carry
with explicit FLOAT32 work, or independent float/integer/[BOOL](../../glossary.md#data-type--datatype) slots, despite their otherwise
valid ordinary [descriptors](../../glossary.md#term-descriptor). Model source validity is not downstream capability.

Source inspection identifies independent guards in ScanBodyInference, ScanGradientRules,
[PrepareContext](../../glossary.md#prepare-context-preparecontext) and CpuGeneralScanLowering; SourceGraphReconstruction and GraphArtifactCodec reject
length mode outright. Direct negative [regression](../../glossary.md#term-regression) tests preserve these boundaries, not relax them.
The former global [dtype](../../glossary.md#data-type--datatype) check implicitly enforced terminal SHARED contribution dtype; Model0045
replaces that implication with an explicit exact check alongside the contribution-[shape](../../glossary.md#shape) check in [Scan](../../glossary.md#general-differentiable-scan).

## Decision drivers

- Keep traversal/[binding](../../glossary.md#term-binding) semantics separate from each ordinary [operation](../../glossary.md#operation)'s [dtype](../../glossary.md#data-type--datatype) and numerics.
- Permit useful typed source descriptions without implying execution or [persistence](../../glossary.md#term-persistence) support.
- Preserve row-role proof, state isolation, exact [carry](../../glossary.md#term-carry) declarations and early failure.
- Keep existing fixed and ordinary [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) APIs, [kernels](../../glossary.md#kernel), [constants](../../glossary.md#term-constant) and [checkpoint](../../glossary.md#term-checkpoint) behavior.

## Options considered

1. Keep the source-wide homogeneous restriction. This would preserve the old implementation but
   prevents valid per-value descriptions and makes source admission mirror CPU qualification.<span data-glossary-navigation="true"> ([homogeneous glossary](../../glossary.md#term-homogeneous); [admission glossary](../../glossary.md#term-admission); [CPU glossary](../../glossary.md#term-cpu); [qualification glossary](../../glossary.md#term-qualification))</span>
2. Relax only Model length-aware source typing and CAST, retaining the closed operation inventory
   and independent downstream guards. This delivers one bounded construction capability.<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [operation glossary](../../glossary.md#operation))</span>
3. Enable typed CPU/AD/cold/NN support together, or add a Scan-wide compute dtype. The former
   requires unselected execution and compatibility decisions; the latter assigns arithmetic
   policy to the loop instead of its ordinary operations. Neither is this task.<span data-glossary-navigation="true"> ([AD glossary](../../glossary.md#autograd); [NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Scan glossary](../../glossary.md#general-differentiable-scan); [dtype glossary](../../glossary.md#data-type--datatype))</span>

## Decision

Select option 2 for the accepted source-only implementation. The exact rules live only in the linked semantic [owner](../../glossary.md#term-owner):
all seven defined [Model](../../glossary.md#term-model) [dtypes](../../glossary.md#data-type--datatype) are admissible per value where existing ordinary operations in
the closed length-aware inventory permit them. Corresponding [bindings](../../glossary.md#term-binding) remain exact; next-[carry](../../glossary.md#term-carry)
[shape](../../glossary.md#shape)/dtype is stable. Valid explicit CAST preserves row and [dependency](../../glossary.md#term-dependency) facts. A terminal SHARED
contribution must explicitly match its own carry dtype as well as shape. [Scan](../../glossary.md#general-differentiable-scan) adds no implicit
conversion, common working [precision](../../glossary.md#term-precision), new [operation](../../glossary.md#operation), type, [signature](../../glossary.md#term-signature) or execution architecture.

## Rationale

For example, a [FLOAT16](../../glossary.md#term-float16) [carry](../../glossary.md#term-carry) can be cast to [FLOAT32](../../glossary.md#term-float32) for ordinary ADD/MUL work and explicitly
cast back to FLOAT16 for the update. The body declares both represented sites. [Scan](../../glossary.md#general-differentiable-scan) only checks
that the returned carry [descriptor](../../glossary.md#term-descriptor) matches the declaration and that row [identity](../../glossary.md#term-identity-meanings) is preserved.
This avoids silently changing [operation](../../glossary.md#operation) semantics or saved-state [rounding](../../glossary.md#term-rounding). Allowing the same
description does not make existing [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) consumers accept a low interior.

## Consequences

### Positive

[Model](../../glossary.md#term-model) construction can represent [heterogeneous](../../glossary.md#term-heterogeneous) roles and explicit working-[precision](../../glossary.md#term-precision) boundaries
without a second [API](../../glossary.md#term-application-programming-interface-api) or a [graph](../../glossary.md#graph)-wide arithmetic setting. Unused discrete slots and [identity](../../glossary.md#term-identity-meanings)
ROW carries no longer require artificial floating conversions.

### Negative and risks

More source [graphs](../../glossary.md#graph) are valid but deliberately un-compilable or un-executable. Documentation
must distinguish those admissions. Global homogeneity previously hid the shared contribution
[dtype](../../glossary.md#data-type--datatype) [invariant](../../glossary.md#term-invariant); omission of the explicit check would admit promoted updates. CAST must not
launder terminal row reductions or shared-[carry](../../glossary.md#term-carry) dependencies into row feedback.

### Migration, testing, and follow-up

[Model0045](../../planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md) coordinates
the exact source/tests/[Javadoc](../../glossary.md#term-javadoc) [scope](../../glossary.md#term-scope) under the accepted review and MAIN GO. Direct raw-attachment,
[graph](../../glossary.md#graph) and capability-query tests prove consumer rejection without relying on earlier guards.
The task's compact Result links the frozen source and [validation](../../glossary.md#term-validation) [artifacts](../../glossary.md#term-artifact), including retained
failed commands and targeted passes; it does not claim a clean full-[module](../../glossary.md#term-module) validation run.

All-valid [Scan](../../glossary.md#general-differentiable-scan), ordinary [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) `.rnn`/`.gru`/`.lstm`, static `Sequence.forward`, fixed [kernels](../../glossary.md#kernel),
old [checkpoints](../../glossary.md#term-checkpoint), [constants](../../glossary.md#term-constant) and numerics remain intact. The eventual direction is [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) recipes over
[general Scan](../../glossary.md#general-differentiable-scan), with no [Model](../../glossary.md#term-model) [RNN](../../glossary.md#term-recurrent-neural-network-rnn) [operation](../../glossary.md#operation); removal or redirection is not selected now.
Higher-order support of unchanged unrolled APIs is not a blocker for this source step.
Generic [CPU](../../glossary.md#term-cpu) typed [bindings](../../glossary.md#term-binding)/[dtypes](../../glossary.md#data-type--datatype), [feature](../../glossary.md#term-feature) [SLICE](../../glossary.md#slice), [layout](../../glossary.md#layout) reads, cold/v5/legacy compatibility and
NN cutover remain coarse follow-up, not dependencies or newly authorized tasks/campaigns.

## Related documentation

- [Architecture explanation and conceptual typed example](../../architecture/general-differentiable-scan.md#per-value-typing-source-construction-only)
- [Foundational operation-local numerical rules](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation))</span>
- [Earlier length-aware decision](0049-length-aware-general-scan.md), retained for its original [scope](../../glossary.md#term-scope)
