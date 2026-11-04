# [Planning](../glossary.md#term-planning)

This directory contains implementation planning documents for Synaptik.

## Choose a plan or a current explanation

A plan records work: its goal, responsible contributor or component, prerequisites, acceptance
conditions and eventual result. It is not a tutorial and does not by itself prove that a backend
can execute a particular operation. If you want to use the framework, start with the
[connected learning project](../tutorials/index.md) or a [focused task guide](../how-to/index.md).

Read an implementation plan in this order:

1. Check its status and prerequisites. **Ready** means work is authorized to start, not that the
   described feature is already implemented.
2. Identify which component calls which API and what responsibility the task changes. The
   [execution explanation](../explanation/packages/execution.md) names the normal framework stages;
   [caller and ownership terms](../glossary.md#caller-callee-and-the-call-boundary) explain the roles.
3. Read the recorded source revision, result and evidence limits. An old successful experiment
   does not qualify a later implementation or a different geometry.
4. For today's usable domain, consult the [current support tables](../backend-guide/production-support.md)
   and the focused reference page, not only a historical task's title.

“Model” in the module list below is the foundational Tensor-description module. A Java
application's owned neural-network Model is a different use of that name; the
[first-model walkthrough](../tutorials/first-model.md) shows the application-facing object.

## Authority boundary

Planning documents are not authoritative architecture contracts. The authoritative [architecture contract](../glossary.md#architecture-contract) is:

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)

If any planning document conflicts with `ARCHITECTURE.md`, `ARCHITECTURE.md` wins and implementation must stop until the conflict is resolved.

## [Planning](../glossary.md#term-planning) guide

- [Planning guide](planning-guide.md)
- [Implementation roadmap](roadmap.md)

## [Module](../glossary.md#term-module) plans

- [Model](modules/model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../glossary.md#term-model))</span> — [capability baseline](modules/model/capabilities.md)
- [Trace](modules/trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../glossary.md#trace))</span>
- [Backend contract](modules/backend-contract/master-plan.md)<span data-glossary-navigation="true"> ([Backend glossary](../glossary.md#backend))</span>
- [Config](modules/config/master-plan.md)
- [Planning](modules/planning/master-plan.md)<span data-glossary-navigation="true"> ([Planning glossary](../glossary.md#term-planning))</span>
- [Compiler](modules/compiler/master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../glossary.md#term-compiler))</span>
- [Runtime](modules/runtime/master-plan.md)<span data-glossary-navigation="true"> ([Runtime glossary](../glossary.md#term-runtime))</span>
- [Prepare](modules/prepare/master-plan.md)<span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare))</span>
- [Engine](modules/engine/master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine))</span>

## [Backend](../glossary.md#backend) plans

- [OpenBLAS provider](backends/openblas-provider/master-plan.md)<span data-glossary-navigation="true"> ([OpenBLAS glossary](../glossary.md#term-openblas))</span>
- [CPU](backends/cpu/master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu))</span>
- [Metal](backends/metal/master-plan.md)<span data-glossary-navigation="true"> ([Metal glossary](../glossary.md#term-metal))</span>
- [CUDA](backends/cuda/master-plan.md)<span data-glossary-navigation="true"> ([CUDA glossary](../glossary.md#term-cuda))</span>

## Extension plans

- [Training](extensions/training/master-plan.md)
- [ONNX](extensions/onnx/master-plan.md)<span data-glossary-navigation="true"> ([ONNX glossary](../glossary.md#term-onnx))</span>

## Tool plans

- [Tuning](tools/tuning/master-plan.md)
- [Benchmarks](tools/benchmarks/master-plan.md)
- [CLI](tools/cli/master-plan.md)<span data-glossary-navigation="true"> ([CLI glossary](../glossary.md#term-cli))</span>
