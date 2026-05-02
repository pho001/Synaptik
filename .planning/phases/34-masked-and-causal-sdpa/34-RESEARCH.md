# Phase 34 Research: Masked And Causal SDPA

## Current Implementation

- Public attention accepts `query`, `key`, `value`, optional `BOOL` mask, and `AttentionOptions` with `causal` and optional scale override.
- `TensorAttentionOps` materializes causal semantics as an effective `BOOL` mask, combines it with an external mask using logical AND, expands it to score shape, and builds one `SCALED_DOT_PRODUCT_ATTENTION` operation with `hasMask=true`.
- CPU execution applies the mask before softmax: mask byte `0` excludes a score using a dtype-specific large negative fill; mask byte non-zero allows the score.
- Metal direct unmasked SDPA is already verified for dense `FLOAT32` rank-3/rank-4 tensors. The native shim builds the primitive DAG as `Q * K^T`, scale multiply, softmax, and `weights * V`.
- Metal direct masked SDPA is currently rejected before lowering with `UNSUPPORTED_MASK_SEMANTICS`, because MPSGraph/native floating mask behavior was not proven equivalent to Synaptik public `BOOL` mask semantics.
- Generic attention-pattern lowering also refuses `WHERE`-masked score graphs, so masked attention does not accidentally become direct native SDPA.

## Gaps To Close

1. Mask semantics need a source-of-truth contract.
   - BOOL mask truth table, causal convention, external+causal AND behavior, scale application order, and rank/broadcast constraints must be locked before admission.
   - Rejection reasons should separate mask dtype, mask shape/broadcast, mask layout, causal unsupported, and backend mask ABI unsupported.

2. The DAG/native contract has no masked SDPA representation.
   - `SDPA` currently uses Q/K/V inputs and rejects a fourth input in native code.
   - A supported path needs either a fourth BOOL/additive mask input contract or a lowered internal DAG that converts BOOL mask semantics before softmax.

3. Causal-only attention has no backend-owned mask generation.
   - The Tensor front-end already creates a causal mask tensor, but supported Metal execution should not require a CPU materialization boundary for that mask on hot paths.
   - The selected representation must make causal handling trace-visible.

4. Layout and dtype legality must be precise.
   - Query/key/value/output remain dense `FLOAT32` in the first executable scope.
   - Mask inputs can be external `BOOL`, Metal-produced `BOOL`, or layout-router repaired dense `BOOL` only when Phase 31/33 contracts prove residency.
   - Non-dense or unsupported masks should reject visibly rather than falling back through tensor-array replay in REQUIRE mode.

5. Coverage needs transformer-specific gates.
   - Existing `transformer_block_hot_path` gates cover unmasked SDPA evidence.
   - Phase 34 needs masked/causal targets requiring native Metal evidence, SDPA mask handling evidence, zero tensor-array fallback, zero CPU fallback, and zero unexpected CPU materialization.

## Recommended Implementation Shape

1. Add a small semantic classifier before execution.
   - Introduce a mask semantic decision helper for direct SDPA that emits stable reason codes/details for:
     - unmasked,
     - external BOOL mask,
     - causal-only effective mask,
     - external AND causal effective mask,
     - unsupported dtype/layout/rank/broadcast.
   - Keep the helper backend-neutral where useful, but bind Metal support claims to Metal capabilities.

2. Extend lowering and ABI deliberately.
   - Prefer extending the `SDPA` DAG node to accept optional input 3 as a BOOL mask with explicit mask mode metadata if the existing node shape fields can carry the necessary contract.
   - If metadata is insufficient, add a small dedicated masked-SDPA node/flag contract rather than overloading scalar bits opaquely.
   - Preserve signature/cache differences between unmasked, masked, causal, and scale variants.

3. Implement native execution with BOOL-to-score semantics.
   - In native MPSGraph, convert the BOOL mask to a score bias or use a verified select/where style graph before softmax.
   - Use the same scale application order as CPU.
   - Avoid public additive-mask API claims unless the implementation explicitly accepts an internal additive representation.

4. Prove parity first on small shapes.
   - Rank-3 and rank-4 cases.
   - external mask, causal-only, external+causal combined mask.
   - broadcasted batch dimensions where current SDPA already supports batch broadcast.
   - rejected dtype/layout/shape cases with stable messages.

5. Add coverage after native parity.
   - Add or extend workloads for masked transformer attention and causal transformer attention.
   - Report mask mode, native buffer evidence, lowered primitive count, CPU exits, tensor-array fallback, and materialization reasons.

## Risks

- Mask polarity is easy to invert. Tests must use asymmetric values where an inverted mask produces clearly different outputs.
- A large negative additive fill can behave differently from CPU for fully masked rows or low precision. First scope should define and test row behavior explicitly.
- Causal masks generated in Java can become CPU leaf tensors unless residency/path handling is deliberate.
- Overloading the existing `SDPA` node without signature metadata can reuse an unmasked compiled executable for a masked graph.
- Supporting external masks without layout repair can reintroduce CPU materialization through mask expansion.

