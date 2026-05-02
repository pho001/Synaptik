# Phase 35 Research: Conv Pool Native Execution

## Current Implementation

- Public Conv2D is NCHW rank-4 with `Conv2dOptions` carrying stride, symmetric padding, dilation, and groups. Optional bias is a rank-1 tensor matching output channels.
- Public pooling is NCHW rank-4 with `Pool2dOptions` carrying kernel, stride, symmetric padding, and avg-pool `countIncludePad`.
- CPU has direct Conv2D and GEMM-lowered Conv2D paths, plus direct max/avg pooling behavior.
- GPU coverage matrix currently marks all conv/pool forward and backward rows unsupported with `CAPABILITY_MISSING` for Metal/CUDA.
- No accelerator DAG node type exists for Conv2D or pool primitives.
- Existing coverage targets already include `conv2d_resnet_3x3` and `max_pool2d_small`, but they expect visible blockers instead of native execution.

## Gaps To Close

1. Conv/pool semantics are not represented in the accelerator DAG.
   - Need node type(s), option metadata encoding, and signature stability.
   - Need to avoid overloading scalar bits beyond what can safely encode stride/pad/dilation/groups/kernel/count flags.

2. Metal planner lacks precise conv/pool legality.
   - Today all conv/pool rows fall through matrix `CAPABILITY_MISSING`.
   - Phase 35 should distinguish supported forward variants from unsupported dtype/layout/rank/groups/dilation/padding/backward variants.

3. Native execution path must match CPU semantics.
   - Conv2D needs NCHW input, OIHW weight, optional bias, stride/padding/dilation/groups.
   - Pooling needs max tie behavior and avg divisor behavior matching CPU.

4. Coverage must flip from blocker to native evidence only for proven scope.
   - `conv2d_resnet_3x3` should require native Metal evidence once Conv2D is supported.
   - `max_pool2d_small` should require native Metal evidence once max pool is supported.
   - Add `avg_pool2d_small` if avg pooling needs separate gate.

## Recommended Implementation Shape

1. Wave 35-01: add contracts and precise rejection.
   - Add semantic helper(s) for Conv2D and pool legality.
   - Keep execution unsupported until native path exists.
   - Add tests for exact reason codes.

2. Wave 35-02: add Conv2D forward.
   - Start with dense `FLOAT32`, rank-4 NCHW, groups=1, dilation=1, symmetric padding, optional bias.
   - Add accelerator DAG node and native MPSGraph implementation or a custom Metal kernel.
   - Prove parity for no-bias and bias cases, plus stride/padding cases.

3. Wave 35-03: add max/avg pool forward.
   - Start with dense `FLOAT32`, rank-4, square or rectangular kernel, stride, optional padding.
   - Be conservative with avg `countIncludePad` if MPSGraph/MPS semantics differ.
   - Add parity tests and rejection tests.

4. Wave 35-04: coverage/docs closure.
   - Convert relevant coverage targets to native gates.
   - Update docs and requirement verification.

## Risks

- Native layout conventions can differ: Synaptik uses NCHW/OIHW, while MPSGraph APIs may prefer NHWC/HWIO or require explicit dimension metadata.
- MPSGraph convolution/group APIs may not match CPU grouped/dilated semantics exactly; unsupported variants must reject visibly.
- Max-pool tie behavior can differ across backends and matters for backward. Forward can be supported without claiming backward tie parity.
- Avg-pool `countIncludePad` is easy to mismatch. If uncertain, support only the divisor mode that matches native behavior and reject the other.
- `CONV2D_GEMM` is CPU-lowered today. Supporting direct `CONV2D` does not automatically support `CONV2D_GEMM`.

