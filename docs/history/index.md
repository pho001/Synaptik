# Historical documentation and [qualification](../glossary.md#term-qualification)

These inventories preserve earlier implementation/qualification descriptions. Their words
“current”, “planned” and “unsupported” describe those earlier slices, not today's product.
They are maintenance evidence, not a learning path or a second [API](../glossary.md#term-application-programming-interface-api) authority.

## Reading the boundaries in these records

An operation being **describable** means an application can construct Tensor metadata for it.
Compiler acceptance means the Compiler can capture and validate that description; backend
qualification means a particular provider can execute the exact type, shape and layout combination.
These are separate claims. A completed task at one boundary does not complete the others.

For example, an application calls Tensor expression methods to describe work and calls Engine to
compile, prepare and run it. A leaf Tensor has no producing expression; it need not already contain
data. Neither that label nor a graph identifier gives the Compiler ownership of application memory.
Read [expression capture](../reference/compiler/capture.md) for how descriptions become graph
values, and [Engine's input-borrowing and result lifecycle](../reference/engine/lifecycle.md)
for when actual storage must stay valid. Use those focused explanations to interpret the vocabulary
below, not to renew an archived execution claim.

## Preserved inventories


| Earlier inventory | Historical record |
|---|---|
| [Tensor](../glossary.md#tensor) introduction and qualification lists | [Qualification overview](reference/tensor/qualification-overview.md) |
| Tensor [compilation](../glossary.md#compiler-engine-and-execution-stage-names)/implementation-stage notes | [Compiler qualification](reference/tensor/compiler-qualification.md) |
| Earlier [Model](../glossary.md#term-model) contract overview | [Model overview](reference/tensor/model-overview.md) |
| Planned/implemented Tensor status inventory | [Historical status](reference/tensor/historical-status.md) |
| [Operation](../glossary.md#operation)-by-operation [Compiler](../glossary.md#term-compiler)/[backend](../glossary.md#backend) slices | [Operation qualification](reference/compiler/operation-qualification.md) |
| Geometry and [normalization](../glossary.md#term-normalization) Compiler/backend slices | [Geometry qualification](reference/compiler/geometry-qualification.md) |

Use the [current reference](../reference/index.md), [support matrix](../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([support matrix glossary](../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-review-matrix))</span>
and [architecture authority index](../../ARCHITECTURE.md) for present decisions. The
[completed audit](../planning/public-release-audit-2026-11-04.md) binds its stated source/[artifacts](../glossary.md#term-artifact);
older receipts elsewhere do not requalify later changes. [Design records](../design/README.md)
and [planning](../planning/README.md)<span data-glossary-navigation="true"> ([planning glossary](../glossary.md#term-planning))</span> remain separate decision and coordination histories.
