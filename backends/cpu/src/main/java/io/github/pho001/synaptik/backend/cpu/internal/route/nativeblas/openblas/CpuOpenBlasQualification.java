package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable evidence that one exact loaded OpenBLAS session passed the bounded CPU qualification.
 * The value retains only immutable compatibility facts and an opaque per-load key; it never owns
 * or retains a provider, coordinator, native segment, or close action. Persistent compatibility
 * is available only for an absolute-path binary whose complete bytes and executable header were
 * inspected. This evidence establishes compatibility only for the target, required symbols,
 * ordinary 32-bit-{@code blasint} contract, and bounded numerical cases recorded here. It does
 * not authenticate the binary or certify arbitrary ABI behavior, numerical behavior, or
 * performance.
 */
public final class CpuOpenBlasQualification {
    /** Current qualification schema. */
    public static final int SCHEMA_VERSION = 1;
    /** Exact bounded numerical-case schema. */
    public static final String NUMERICAL_CASE_VERSION =
            "SYNAPTIK_OPENBLAS_GEMM_QUALIFICATION_V1";
    /** Required symbols in provider ABI order. */
    public static final List<String> REQUIRED_SYMBOLS = List.of(
            "cblas_sgemm", "cblas_dgemm", "openblas_set_num_threads",
            "openblas_get_num_threads");

    /** Lifetime scope of the issued evidence. */
    public enum Scope {
        /** Complete path-loaded binary facts may be compared for later persistent reuse. */
        PERSISTENT_BINARY,
        /** Evidence is valid only with the exact live coordinator that issued it. */
        SESSION_ONLY
    }
    /** Supported host operating-system families. */
    public enum OperatingSystem {
        /** Apple macOS. */ MACOS,
        /** Linux. */ LINUX,
        /** Microsoft Windows. */ WINDOWS
    }
    /** Supported canonical 64-bit machines. */
    public enum Machine {
        /** 64-bit Arm. */ AARCH64,
        /** 64-bit x86. */ X86_64
    }
    /** Supported executable containers. */
    public enum ExecutableFormat {
        /** Thin 64-bit Mach-O. */ MACH_O_64,
        /** Little-endian ELF64. */ ELF_64,
        /** PE32+ with a 64-bit supported machine. */ PE_32_PLUS
    }
    /** Locked provider integer ABI evidence. */
    public enum BlasIntAbi {
        /** Every provider integer parameter uses an ordinary 32-bit C {@code int}. */ C_INT_32
    }

    /**
     * Stable target facts relevant to the current ordinary C ABI.
     *
     * @param schemaVersion target schema, currently one
     * @param operatingSystem supported operating-system family
     * @param machine canonical host machine
     * @param addressWidthBits native address width, exactly 64
     * @param byteOrder native byte order, exactly little endian
     */
    public record TargetFingerprint(int schemaVersion, OperatingSystem operatingSystem,
            Machine machine, int addressWidthBits, ByteOrder byteOrder) {
        /**
         * Validates the closed supported target.
         *
         * @throws NullPointerException if an enum or byte-order value is {@code null}
         * @throws IllegalArgumentException if the schema, address width, or byte order is not
         *     supported
         */
        public TargetFingerprint {
            Objects.requireNonNull(operatingSystem, "operatingSystem");
            Objects.requireNonNull(machine, "machine");
            Objects.requireNonNull(byteOrder, "byteOrder");
            if (schemaVersion != 1 || addressWidthBits != 64
                    || byteOrder != ByteOrder.LITTLE_ENDIAN) {
                throw new IllegalArgumentException("unsupported OpenBLAS target fingerprint");
            }
        }
    }

    /**
     * Stable identity of the complete inspected binary content.
     *
     * @param schemaVersion binary schema, currently one
     * @param digestAlgorithm digest algorithm, exactly {@code SHA-256}
     * @param sha256 lowercase 64-character hexadecimal complete-file digest
     * @param byteLength positive complete file length, at most one GiB
     * @param executableFormat parsed 64-bit executable format
     * @param machine parsed canonical machine
     */
    public record BinaryIdentity(int schemaVersion, String digestAlgorithm, String sha256,
            long byteLength, ExecutableFormat executableFormat, Machine machine) {
        /**
         * Validates the exact versioned content identity.
         *
         * @throws NullPointerException if an algorithm, digest, format, or machine is
         *     {@code null}
         * @throws IllegalArgumentException if a schema, algorithm, digest, length, format, or
         *     machine value is outside the closed identity contract
         */
        public BinaryIdentity {
            Objects.requireNonNull(digestAlgorithm, "digestAlgorithm");
            Objects.requireNonNull(sha256, "sha256");
            Objects.requireNonNull(executableFormat, "executableFormat");
            Objects.requireNonNull(machine, "machine");
            if (schemaVersion != 1 || !digestAlgorithm.equals("SHA-256")
                    || !sha256.matches("[0-9a-f]{64}") || byteLength <= 0
                    || byteLength > CpuOpenBlasBinaryInspector.MAXIMUM_BINARY_BYTES) {
                throw new IllegalArgumentException("invalid OpenBLAS binary identity");
            }
        }
    }

    /**
     * Complete equality projection suitable as a later persistent compatibility input. The
     * projection excludes the live-session key and diagnostic path and file-attribute facts.
     * Equality indicates matching versioned compatibility inputs, not binary authentication.
     *
     * @param qualificationSchemaVersion qualification schema, currently one
     * @param targetFingerprint exact supported target facts
     * @param requiredSymbols exact ordered four-symbol inventory
     * @param blasIntAbi exact ordinary provider integer ABI evidence
     * @param numericalCaseVersion exact bounded numerical-case version
     * @param binaryIdentity complete inspected binary content identity
     */
    public record PersistentIdentity(int qualificationSchemaVersion,
            TargetFingerprint targetFingerprint, List<String> requiredSymbols,
            BlasIntAbi blasIntAbi, String numericalCaseVersion, BinaryIdentity binaryIdentity) {
        /**
         * Defensively snapshots every versioned compatibility fact.
         *
         * @throws NullPointerException if a required fact or symbol-list entry is {@code null}
         * @throws IllegalArgumentException if the schema, required-symbol inventory, ABI, or
         *     numerical-case version is not the exact supported value
         */
        public PersistentIdentity {
            Objects.requireNonNull(targetFingerprint, "targetFingerprint");
            requiredSymbols = List.copyOf(requiredSymbols);
            Objects.requireNonNull(blasIntAbi, "blasIntAbi");
            Objects.requireNonNull(numericalCaseVersion, "numericalCaseVersion");
            Objects.requireNonNull(binaryIdentity, "binaryIdentity");
            if (qualificationSchemaVersion != SCHEMA_VERSION
                    || !requiredSymbols.equals(REQUIRED_SYMBOLS)
                    || blasIntAbi != BlasIntAbi.C_INT_32
                    || !numericalCaseVersion.equals(NUMERICAL_CASE_VERSION)) {
                throw new IllegalArgumentException("invalid persistent qualification facts");
            }
        }
    }

    /** Stable unchecked failure for an ordinary qualification rejection. */
    public static final class QualificationException extends IllegalStateException {
        /**
         * Creates a rejection retaining its original cause.
         * @param message stable non-null diagnostic message
         * @param cause original failure, possibly {@code null}
         */
        QualificationException(String message, Throwable cause) { super(message, cause); }
    }

    /** Opaque identity-only association created once for each loaded provider resource. */
    static final class SessionKey {
        /** Creates one fresh key whose object identity is its only meaning. */
        SessionKey() { }
    }

    private final Scope scope;
    private final TargetFingerprint targetFingerprint;
    private final List<String> requiredSymbols;
    private final BlasIntAbi blasIntAbi;
    private final String numericalCaseVersion;
    private final Optional<BinaryIdentity> binaryIdentity;
    private final SessionKey sessionKey;

    /**
     * Creates one successful credential from issuer-controlled facts.
     *
     * @param scope exact lifetime and persistence scope
     * @param targetFingerprint exact supported target facts
     * @param binaryIdentity binary identity present exactly for persistent-binary scope
     * @param sessionKey opaque key from the exact loaded resource
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if scope and binary-identity presence disagree
     */
    CpuOpenBlasQualification(Scope scope, TargetFingerprint targetFingerprint,
            Optional<BinaryIdentity> binaryIdentity, SessionKey sessionKey) {
        this.scope = Objects.requireNonNull(scope, "scope");
        this.targetFingerprint = Objects.requireNonNull(targetFingerprint, "targetFingerprint");
        this.requiredSymbols = REQUIRED_SYMBOLS;
        this.blasIntAbi = BlasIntAbi.C_INT_32;
        this.numericalCaseVersion = NUMERICAL_CASE_VERSION;
        this.binaryIdentity = Objects.requireNonNull(binaryIdentity, "binaryIdentity");
        this.sessionKey = Objects.requireNonNull(sessionKey, "sessionKey");
        if ((scope == Scope.PERSISTENT_BINARY) != binaryIdentity.isPresent()) {
            throw new IllegalArgumentException("qualification scope and binary identity disagree");
        }
    }

    /** Returns the qualification schema version.
     * @return qualification schema version one */
    public int schemaVersion() { return SCHEMA_VERSION; }
    /** Returns the credential lifetime and persistence scope.
     * @return exact immutable qualification scope; never {@code null} */
    public Scope scope() { return scope; }
    /** Returns the qualified host target facts.
     * @return exact immutable target fingerprint; never {@code null} */
    public TargetFingerprint targetFingerprint() { return targetFingerprint; }
    /** Returns the symbol inventory exercised by the qualification contract.
     * @return exact immutable ordered required-symbol inventory; never {@code null} */
    public List<String> requiredSymbols() { return requiredSymbols; }
    /** Returns the provider integer ABI assumed and exercised by bounded checks.
     * @return locked ordinary 32-bit C integer ABI evidence; never {@code null} */
    public BlasIntAbi blasIntAbi() { return blasIntAbi; }
    /** Returns the version of the bounded numerical cases that were executed.
     * @return exact bounded numerical-case schema; never {@code null} */
    public String numericalCaseVersion() { return numericalCaseVersion; }
    /** Returns the complete-file identity when exact path inspection was possible.
     * @return non-null binary identity optional, present exactly for persistent-binary scope */
    public Optional<BinaryIdentity> binaryIdentity() { return binaryIdentity; }
    /**
     * Projects all stable compatibility facts without exposing the live-session key.
     * @return non-null complete persistent identity, or empty for a name-loaded session
     */
    public Optional<PersistentIdentity> persistentIdentity() {
        return binaryIdentity.map(binary -> new PersistentIdentity(SCHEMA_VERSION,
                targetFingerprint, requiredSymbols, blasIntAbi, numericalCaseVersion, binary));
    }

    /**
     * Tests exact live-session association by key identity.
     *
     * @param candidate coordinator key to compare; may be {@code null}
     * @return whether {@code candidate} is the exact key retained by this credential
     */
    boolean belongsTo(SessionKey candidate) { return sessionKey == candidate; }
}
