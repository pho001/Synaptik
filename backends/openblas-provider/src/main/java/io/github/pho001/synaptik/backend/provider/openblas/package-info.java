/**
 * Provides explicit caller-directed OpenBLAS loading and low-level native bindings.
 *
 * <p>The package is a JDK-only leaf below the CPU backend. It owns library lookup lifetime and
 * exact OpenBLAS C symbol binding, while CPU policy owns route selection, fallback, configuration,
 * and thread choices. Current public API performs no native operation.
 */
package io.github.pho001.synaptik.backend.provider.openblas;
