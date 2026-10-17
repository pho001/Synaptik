/**
 * Public backend-neutral training lifecycle over generic live parameters and the ordinary Engine
 * execution facade.
 *
 * <p>{@link io.github.pho001.synaptik.training.TrainingSession} compiles and prepares one scalar
 * forward/backward graph at construction, reuses it across runs, validates Compiler-authoritative
 * input and gradient occurrences, and applies session-owned optimizer state to stable
 * caller-owned host parameter storage. The package neither implements autograd nor imports
 * NN, Runtime, Prepare, tuning, or a concrete backend. A
 * {@link io.github.pho001.synaptik.training.TrainingParameters} source retains its live owner and
 * validates exact ordered membership; Training separately owns storage and value validation.</p>
 */
package io.github.pho001.synaptik.training;
