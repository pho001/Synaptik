# [API](../../glossary.md#term-application-programming-interface-api) and [Javadoc](../../glossary.md#term-javadoc) style

## Purpose

[API](../../glossary.md#term-application-programming-interface-api) documentation and [Javadoc](../../glossary.md#term-javadoc) define contracts that callers can rely on without reading implementation source. They explain semantics, valid use, observable behavior, and failures precisely.

An **[application programming interface](../../glossary.md#api-spi-and-dto) (API)** is the set of types and operations that callers use.
**Javadoc** turns specially structured Java comments into browsable reference pages. Document what
an [operation](../../glossary.md#operation) means, not just how to call it: **semantics** means its behavior, an **[invariant](../../glossary.md#term-invariant)** is a
condition that must remain true, and **[nullability](../../glossary.md#term-nullability)** says whether a value may be absent as `null`.

Do not confuse an API with a **[service provider interface](../../glossary.md#term-service-provider-interface-spi) ([SPI](../../glossary.md#api-spi-and-dto))**, a contract implemented by a
component supplying a service, or an **[application binary interface](../../glossary.md#term-application-binary-interface-abi) ([ABI](../../glossary.md#application-binary-interface-and-native-bridge))**, the compiled calling
and data-[layout](../../glossary.md#layout) agreement. [Backend](../../glossary.md#backend) implementers may implement an SPI; a [native bridge](../../glossary.md#term-native-bridge) must also
respect its ABI. Describing those boundaries does not make an internal interface a public API.

Apply [General style](general-style.md) with this profile. The mandatory Java documentation rules in [`AGENTS.md`](../../../AGENTS.md) still apply.

## Required content

For each affected public or contract-relevant type or member, document as applicable:

- purpose, semantics, and intended use;
- [parameter](../../glossary.md#term-parameter) meaning, constraints, units, [nullability](../../glossary.md#term-nullability), [ownership](../../glossary.md#term-ownership-meanings), and mutation behavior;
- return meaning, nullability, [identity](../../glossary.md#term-identity-meanings), mutability, and ownership;
- expected exceptions and the conditions that cause them;
- invariants, [lifecycle](../../glossary.md#lifecycle), threading, equality, [resource lifetime](../../glossary.md#ownership-borrowing-and-lease), and side effects;
- boundaries and deliberately unsupported behavior; and
- a focused example when the contract or interaction is not obvious.

Every constructor and method parameter requires `@param`. Every non-`void` method requires `@return`. Expected caller-visible failures require `@throws`. Constructors and `void` methods do not use `@return`.

[API](../../glossary.md#term-application-programming-interface-api) reference pages should connect related types into a usable [mental model](../../glossary.md#term-mental-model) rather than reproduce generated [signatures](../../glossary.md#term-signature). Examples must state inputs, output, and interpretation. Use the [example format](example-format.md) for multi-step examples.

## Make the call boundary readable

Name the concrete caller and callee before writing “caller-owned.” For example,
when an application passes an Arena to a parameter-copy utility, say that the
utility copies parameter bytes into storage controlled by that Arena, while the
application must keep the Arena alive for the consuming session. Do not generalize
this into “every Tensor call copies” or “every Tensor must be closed.”

When documenting equality, distinguish Java's [`hashCode`](../../glossary.md#term-hash-code)
(collection bucketing consistent with equality) from a cryptographic content digest
such as SHA-256. Equal hash codes do not prove equal objects, equal bytes or an
authentic publisher. A resource owner can intentionally use reference identity
even when two logical recipes describe equal computation.

For an internal SPI, Runtime can be the caller and a backend implementation the
callee. Explain which prepared descriptions cross first, which live storage is
borrowed later and where cleanup remains. The
[Runtime/Prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md)
provides a concrete multi-component example; [example format](example-format.md)
turns its contracts into an interpretable procedure.

## Avoid

- restating a method name without defining behavior;
- documenting implementation strategy as a stable contract;
- vague phrases such as “returns the result”;
- missing units, [nullability](../../glossary.md#term-nullability), [ownership](../../glossary.md#term-ownership-meanings), or mutation semantics;
- exceptions listed without trigger conditions;
- examples that omit [validation](../../glossary.md#term-validation) or failure behavior when it matters; and
- [Javadoc](../../glossary.md#term-javadoc) that contradicts tests or the public reference.

## [Validation](../../glossary.md#term-validation)

- Generate [Javadoc](../../glossary.md#term-javadoc) for every affected [module](../../glossary.md#term-module) after final Javadoc edits. Reuse recorded module-test evidence unless executable Java behavior changed afterward.
- Compare contract claims with [signatures](../../glossary.md#term-signature), implementation, and focused tests.
- Check that each [parameter](../../glossary.md#term-parameter), non-`void` return, and expected failure is documented.
- Verify links and rendered code blocks.
- Confirm public reference pages and Javadoc use the same terminology and boundaries.

## [Javadoc](../../glossary.md#term-javadoc) template

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

## [API](../../glossary.md#term-application-programming-interface-api) reference template

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
