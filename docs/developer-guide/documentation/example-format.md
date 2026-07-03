# Example format

## Purpose

An example should let a newcomer connect an input to an observable result and explain why the result matters. Use this format in developer, user, API, and backend documentation at the level of completeness appropriate to the document type.

## Required structure

For a substantive example, provide:

1. **Goal:** what the example demonstrates.
2. **Prerequisites and initial state:** what already exists or must be configured.
3. **Concrete inputs:** exact values, types, shapes, files, or configuration.
4. **Code or procedure:** a complete path through the concept.
5. **Meaningful-step explanation:** explain lines or groups that change semantics, state, ownership, or output; do not narrate punctuation or boilerplate.
6. **Intermediate results:** values or state that help the reader verify progress.
7. **Final result:** exact output or observable state.
8. **Interpretation:** what the result proves and what it does not prove.
9. **Failure or variation:** include one when it teaches an important constraint.

If the example uses conceptual pseudocode or a planned API, label it before the code block. A user-guide command presented as runnable must actually run in its stated environment.

## Numerical example: shape element count

### Goal

Show how a fully static shape determines its logical element count.

### Inputs

The three axis sizes are `2`, `3`, and `4`. All are static, non-negative dimensions.

```java
import io.github.pho001.synaptik.model.shape.Shape;

Shape shape = Shape.of(2, 3, 4);
long elementCount = shape.knownElementCount().orElseThrow();
```

### Meaningful lines

- `Shape.of(2, 3, 4)` creates a rank-3 shape. The axes have sizes 2, 3, and 4 in that order.
- `knownElementCount()` asks for the product of all static dimensions. `orElseThrow()` is safe in this example because none of the dimensions is dynamic.

The calculation is:

```text
2 × 3 × 4 = 24
```

An explicit intermediate walkthrough is:

```text
after axes 0 and 1: 2 × 3 = 6
after axis 2:       6 × 4 = 24
```

### Result and interpretation

`elementCount` is `24`. The result means that a tensor with this fully static shape has 24 logical element positions. It does not allocate storage, choose a layout, or say which backend will execute an operation on the tensor.

### Useful variation

If any dimension is dynamic, `knownElementCount()` is empty because the numeric product is not yet known. The example must not replace that unknown size with a negative sentinel.

## Nonmathematical example: validating a documentation link

### Goal

Show the same input-to-result structure for a workflow that does not involve data transformation, without inventing arithmetic.

### Initial state and input

A developer guide contains this relative link:

```markdown
[Planning Guide](../../planning/planning-guide.md)
```

The input is the source document path plus the relative target. The expected target is the repository's planning guide.

### Procedure and intermediate result

Run the repository's Markdown link check, or a targeted local path-and-anchor check when no repository checker exists. The checker resolves the link relative to the source file, then verifies that the target file exists.

The intermediate result is the resolved repository path `docs/planning/planning-guide.md`.

### Result and interpretation

The check passes with no missing-target error. This proves that the link resolves in the checked repository state. It does not prove that the target content is accurate, so the documentation review must still inspect meaning and authority.

## Avoid

- placeholder inputs such as `foo` when a domain value would teach more;
- output shown without explaining how it follows from the input;
- snippets that omit required setup while claiming to be complete;
- mechanical commentary on every brace or import;
- unexplained numerical results;
- forced calculations for nonmathematical workflows; and
- conceptual code presented as a supported API.

## Validation

- Run or compile examples when their document type promises runnable code.
- Recalculate every numerical walkthrough independently.
- Verify intermediate and final results against current behavior or tests.
- Confirm each meaningful code line or procedural step is explained.
- Check that interpretation distinguishes what the result proves from adjacent concerns.
- Validate links and clearly label conceptual examples.
