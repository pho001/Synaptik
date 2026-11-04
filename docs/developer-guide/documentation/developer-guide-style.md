# Developer guide style

## Purpose

A developer guide teaches a contributor how a Synaptik concept or internal workflow works well enough to use, debug, or extend it safely. It is instructional, not merely a catalog of classes.

Apply [General style](general-style.md) and the [example format](example-format.md) alongside this profile.

A **[mental model](../../glossary.md#term-mental-model)** is a small explanation of how the important parts relate. A **prerequisite**
is something the reader needs before following the guide. A **[lifecycle](../../glossary.md#lifecycle)** traces creation, use
and [release](../../glossary.md#term-release); an **[invariant](../../glossary.md#term-invariant)** states what must remain true during that sequence. Teach these
relationships with a concrete case rather than requiring the reader to infer them from identifiers.

## Required content

Every substantive developer guide must include:

- **What the reader will learn:** concrete capabilities or understanding gained.
- **Prerequisites:** required repository knowledge, tools, completed setup, and linked background.
- **Terms:** short definitions for the concepts needed in the guide, with glossary links.
- **[Mental model](../../glossary.md#term-mental-model):** the components, [ownership](../../glossary.md#term-ownership-meanings), sequence, or data flow before implementation detail.
- **Complete example:** enough context to follow the whole concept, not an isolated unexplained fragment.
- **Meaningful-line explanation:** explain every line or group of lines that changes meaning, state, ownership, or output. Do not mechanically narrate braces, imports, or repeated syntax.
- **Concrete inputs:** exact values, configuration, initial state, and assumptions.
- **Intermediate results:** important state or values after each meaningful transformation.
- **Final result and interpretation:** what was produced and what the result tells the reader.
- **Typical mistakes:** likely misconceptions, symptoms, causes, and corrections.
- **Limitations:** unsupported cases, boundaries, [lifecycle](../../glossary.md#lifecycle) constraints, and deferred behavior.
- **Related links:** authoritative architecture, [API](../../glossary.md#term-application-programming-interface-api) contracts, glossary entries, and adjacent guides.

When the concept performs a calculation or transforms data, include a concrete numerical walkthrough. Show the input values, calculation or transformation, intermediate values, final value, and interpretation. Do not force arithmetic into topics where it adds no understanding, such as repository navigation or a decision-record workflow; use a concrete nonmathematical scenario instead.

## Explain the tool chain through one contributor action

For an architecture-check command, identify the contributor as the shell user,
the Gradle wrapper as the task launcher, and the architecture-test project as the
assertion owner. Explain which repository relationships it reads and what a failure
means. A test result covers those assertions, not every rule in the contract.
[Architecture tests](../architecture-tests.md) is a focused example of that procedure.

For code examples, place short why-comments inside the fence at storage borrowing,
copying or cleanup boundaries. Keep executable tokens unchanged when revising only
an explanation. Use [example format](example-format.md) for result interpretation
and [General style](general-style.md#write-mathematics-as-mathematics) for calculations
with defined symbols, units and valid domains.

## Avoid

- source-code tours that list files without explaining responsibility;
- fragments that omit setup, inputs, output, or interpretation;
- line-by-line narration of syntax with no semantic effect;
- replacing the [architecture contract](../../glossary.md#architecture-contract) with tutorial prose;
- presenting a future [API](../../glossary.md#term-application-programming-interface-api) as implemented;
- hiding failure paths or prerequisites; and
- artificial numerical examples for concepts that do not involve computation or data transformation.

## [Validation](../../glossary.md#term-validation)

- Follow one newcomer through the guide from prerequisites to interpreted result.
- Verify the example against current code and tests, or label it conceptual.
- Confirm every meaningful example line is explained.
- For computation or data transformation, recalculate the numerical walkthrough independently.
- Check that mistakes and limitations correspond to real boundaries.
- Validate all links and glossary references.

## Template

```markdown
# <Concept or workflow>

## What you will learn

## Prerequisites

## Terms

## Mental model

## Complete example

### Inputs and initial state

### Code or procedure

### What each meaningful line does

### Intermediate results

### Final result and interpretation

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|

## Limitations and boundaries

## Related documentation
```
