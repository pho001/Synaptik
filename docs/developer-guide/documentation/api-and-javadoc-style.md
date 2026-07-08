# API and Javadoc style

## Purpose

API documentation and Javadoc define contracts that callers can rely on without reading implementation source. They explain semantics, valid use, observable behavior, and failures precisely.

Apply [General style](general-style.md) with this profile. The mandatory Java documentation rules in [`AGENTS.md`](../../../AGENTS.md) still apply.

## Required content

For each affected public or contract-relevant type or member, document as applicable:

- purpose, semantics, and intended use;
- parameter meaning, constraints, units, nullability, ownership, and mutation behavior;
- return meaning, nullability, identity, mutability, and ownership;
- expected exceptions and the conditions that cause them;
- invariants, lifecycle, threading, equality, resource lifetime, and side effects;
- boundaries and deliberately unsupported behavior; and
- a focused example when the contract or interaction is not obvious.

Every constructor and method parameter requires `@param`. Every non-`void` method requires `@return`. Expected caller-visible failures require `@throws`. Constructors and `void` methods do not use `@return`.

API reference pages should connect related types into a usable mental model rather than reproduce generated signatures. Examples must state inputs, output, and interpretation. Use the [example format](example-format.md) for multi-step examples.

## Avoid

- restating a method name without defining behavior;
- documenting implementation strategy as a stable contract;
- vague phrases such as “returns the result”;
- missing units, nullability, ownership, or mutation semantics;
- exceptions listed without trigger conditions;
- examples that omit validation or failure behavior when it matters; and
- Javadoc that contradicts tests or the public reference.

## Validation

- Generate Javadoc for every affected module after final Javadoc edits. Reuse recorded module-test evidence unless executable Java behavior changed afterward.
- Compare contract claims with signatures, implementation, and focused tests.
- Check that each parameter, non-`void` return, and expected failure is documented.
- Verify links and rendered code blocks.
- Confirm public reference pages and Javadoc use the same terminology and boundaries.

## Javadoc template

```java
/**
 * Explains what the operation means, when to use it, and its key invariant.
 *
 * @param input the input value; must be non-null and satisfy ...
 * @param axis the zero-based axis; must be in ...
 * @return the immutable result ...; never {@code null}
 * @throws NullPointerException if {@code input} is {@code null}
 * @throws IllegalArgumentException if {@code axis} is outside ...
 */
Result transform(Input input, int axis);
```

## API reference template

```markdown
# <API area>

## Purpose and mental model

## Core contracts

## Inputs, outputs, and ownership

## Lifecycle and invariants

## Complete example

## Failures and limitations

## Related API and architecture
```
