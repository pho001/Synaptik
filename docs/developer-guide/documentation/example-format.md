# Example format

## Purpose

An example should let a newcomer connect an input to an observable result and explain why the result matters. Use this format in developer, user, [API](../../glossary.md#term-application-programming-interface-api), and [backend](../../glossary.md#backend) documentation at the level of completeness appropriate to the document type.

**[Pseudocode](../../glossary.md#term-pseudocode)** illustrates an algorithm without promising executable syntax; a **runnable example**
uses the real interface and includes the setup needed to execute it. A test **[fixture](../../glossary.md#term-fixture)** is the
prepared data and environment for a repeatable test. A **[mock](../../glossary.md#term-mock)** substitutes controlled behavior
for a collaborator. If an example uses either, name what is supplied or substituted so the reader
does not mistake the demonstration for evidence about a real backend or installation.

## Required structure

For a substantive example, provide:

1. **Goal:** what the example demonstrates.
2. **Prerequisites and initial state:** what already exists or must be configured.
3. **Concrete inputs:** exact values, types, [shapes](../../glossary.md#shape), files, or configuration.
4. **Code or procedure:** a complete path through the concept.
5. **Meaningful-step explanation:** explain lines or groups that change semantics, state, [ownership](../../glossary.md#term-ownership-meanings), or output; do not narrate punctuation or boilerplate.
6. **Intermediate results:** values or state that help the reader verify progress.
7. **Final result:** exact output or observable state.
8. **Interpretation:** what the result proves and what it does not prove.
9. **Failure or variation:** include one when it teaches an important constraint.

If the example uses conceptual [pseudocode](../../glossary.md#term-pseudocode) or a planned [API](../../glossary.md#term-application-programming-interface-api), label it before the code block. A user-guide command presented as runnable must actually run in its stated environment.

## Numerical example: [shape](../../glossary.md#shape) [element](../../glossary.md#term-element) count

### Goal

Show how a fully static [shape](../../glossary.md#shape) determines its logical [element](../../glossary.md#term-element) count.

### Inputs

The three [axis](../../glossary.md#term-axis) sizes are `2`, `3`, and `4`. All are static, non-negative [dimensions](../../glossary.md#dimension).

```java
import io.github.pho001.synaptik.model.shape.Shape;

// A Shape describes logical positions; this does not allocate Tensor storage.
Shape shape = Shape.of(2, 3, 4);
// Every extent is static here, so the optional count is present.
long elementCount = shape.knownElementCount().orElseThrow();
```

### Meaningful lines

- `Shape.of(2, 3, 4)` creates a [rank](../../glossary.md#term-rank)-3 [shape](../../glossary.md#shape). The [axes](../../glossary.md#term-axis) have sizes 2, 3, and 4 in that order.
- `knownElementCount()` asks for the product of all static [dimensions](../../glossary.md#dimension). `orElseThrow()` is safe in this example because none of the dimensions is dynamic.

Let $r$ be the nonnegative integer rank and $d_k$ the nonnegative integer extent
of axis $k$, with $0\le k<r$. For fully static extents, the logical count is

$$
E=\prod_{k=0}^{r-1}d_k.
$$

Here $r=3$ and $(d_0,d_1,d_2)=(2,3,4)$, so $E=(2\cdot3)\cdot4=6\cdot4=24$
logical elements. The first two axes account for six coordinate pairs; the final
axis supplies four positions for each pair. This is exact integer geometry, not
floating-point arithmetic or a byte-allocation calculation. The rank-zero empty
product is one; any known zero extent makes the product zero. The API still uses
checked representable counts rather than promising arbitrarily large Java arrays.

### Result and interpretation

`elementCount` is `24`. The result means that a [tensor](../../glossary.md#tensor) with this fully static [shape](../../glossary.md#shape) has 24 logical [element](../../glossary.md#term-element) positions. It does not allocate storage, choose a [layout](../../glossary.md#layout), or say which [backend](../../glossary.md#backend) will execute an [operation](../../glossary.md#operation) on the tensor.

### Useful variation

If any [dimension](../../glossary.md#dimension) is dynamic, `knownElementCount()` is empty because the numeric product is not yet known. The example must not replace that unknown size with a negative sentinel.

## Nonmathematical example: validating a documentation link

### Goal

Show the same input-to-result structure for a workflow that does not involve data transformation, without inventing arithmetic.

### Initial state and input

A developer guide contains this relative link:

```markdown
[Planning Guide](../../planning/planning-guide.md)
```

The input is the source document path plus the relative [target](../../glossary.md#term-target). The expected target is the repository's [planning](../../glossary.md#term-planning) guide.

### Procedure and intermediate result

Run the repository's Markdown link check, or a targeted local path-and-anchor check when no repository checker exists. The checker resolves the link relative to the source file, then verifies that the [target](../../glossary.md#term-target) file exists.

The intermediate result is the resolved repository path `docs/planning/planning-guide.md`.

### Result and interpretation

The check passes with no missing-[target](../../glossary.md#term-target) error. This proves that the link resolves in the checked repository state. It does not prove that the target content is accurate, so the documentation review must still inspect meaning and authority.

## Relate inputs to the API and to the teaching claim

In the shape example, the contributor calls Model's Shape API with extents and
receives metadata. No Tensor values cross that boundary, so neither copying values
nor resource cleanup is part of the result. In the link example, the contributor
supplies a source path and a target; the checker reports resolution, not semantic
accuracy.

Use [shape reference](../../reference/tensor/shapes.md) for geometry details and
[API/Javadoc style](api-and-javadoc-style.md#make-the-call-boundary-readable) when
an example does borrow or copy live values. Place a why-comment at that boundary,
not in literal stdout, strict JSON or archived proof bytes.

## Avoid

- placeholder inputs such as `foo` when a domain value would teach more;
- output shown without explaining how it follows from the input;
- snippets that omit required setup while claiming to be complete;
- mechanical commentary on every brace or import;
- unexplained numerical results;
- forced calculations for nonmathematical workflows; and
- conceptual code presented as a supported [API](../../glossary.md#term-application-programming-interface-api).

## [Validation](../../glossary.md#term-validation)

- Run or compile examples when their document type promises runnable code.
- Recalculate every numerical walkthrough independently.
- Verify intermediate and final results against current behavior or tests.
- Confirm each meaningful code line or procedural step is explained.
- Check that interpretation distinguishes what the result proves from adjacent concerns.
- Validate links and clearly label conceptual examples.
