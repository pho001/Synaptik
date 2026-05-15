package backend.metal.bridge;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Locale;

/**
 * Resolves the native Metal MPS shim used by all Java FFM Metal bridges.
 *
 * <p>Explicit operator configuration always wins. When no explicit path is configured, the resolver
 * extracts the platform-specific shim bundled in the Synaptik JAR into a content-addressed local cache
 * and loads that extracted file. The final fallback is the platform library name, which keeps local
 * development workflows with manually installed libraries working.</p>
 */
public final class MetalNativeLibraryResolver {
    public static final String PROPERTY = "synaptik.metal.mps.lib";
    public static final String ENVIRONMENT = "SYNAPTIK_METAL_MPS_LIB";
    public static final String CACHE_DIR_PROPERTY = "synaptik.native.cache.dir";
    public static final String LIBRARY_NAME = "synaptik_apple_mps";
    public static final String LIBRARY_FILE_NAME = "libsynaptik_apple_mps.dylib";

    private MetalNativeLibraryResolver() {
    }

    /**
     * Resolves a symbol lookup for the Metal MPS shim.
     *
     * @param arena arena that owns the native library lookup
     * @return symbol lookup for the configured, bundled, or system Metal shim
     */
    public static SymbolLookup resolveLookup(Arena arena) {
        String explicit = System.getProperty(PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return SymbolLookup.libraryLookup(explicit.trim(), arena);
        }
        String envLib = System.getenv(ENVIRONMENT);
        if (envLib != null && !envLib.isBlank()) {
            return SymbolLookup.libraryLookup(envLib.trim(), arena);
        }

        Throwable bundledFailure = null;
        try {
            Path bundled = extractBundledLibrary();
            if (bundled != null) {
                return SymbolLookup.libraryLookup(bundled.toString(), arena);
            }
        } catch (Throwable t) {
            bundledFailure = t;
        }

        try {
            return SymbolLookup.libraryLookup(LIBRARY_NAME, arena);
        } catch (Throwable systemFailure) {
            if (bundledFailure == null) {
                throw systemFailure;
            }
            IllegalStateException combined = new IllegalStateException(
                    "Metal MPS lookup failed for bundled shim and system library. Bundled: "
                            + bundledFailure.getClass().getSimpleName() + ": " + safeMessage(bundledFailure)
                            + "; system: " + systemFailure.getClass().getSimpleName() + ": " + safeMessage(systemFailure),
                    systemFailure
            );
            combined.addSuppressed(bundledFailure);
            throw combined;
        }
    }

    static Path extractBundledLibrary() throws IOException {
        String resourceName = bundledResourceName();
        ClassLoader loader = MetalNativeLibraryResolver.class.getClassLoader();
        byte[] bytes;
        try (InputStream in = loader == null
                ? ClassLoader.getSystemResourceAsStream(resourceName)
                : loader.getResourceAsStream(resourceName)) {
            if (in == null) {
                return null;
            }
            bytes = in.readAllBytes();
        }

        String digest = sha256(bytes);
        Path targetDir = cacheRoot()
                .resolve("metal-mps")
                .resolve(currentPlatformId())
                .resolve(digest);
        Files.createDirectories(targetDir);
        Path target = targetDir.resolve(LIBRARY_FILE_NAME);
        if (!Files.exists(target) || Files.size(target) != bytes.length) {
            Path tmp = Files.createTempFile(targetDir, LIBRARY_FILE_NAME, ".tmp");
            Files.write(tmp, bytes);
            tmp.toFile().setReadable(true, false);
            tmp.toFile().setExecutable(true, false);
            try {
                Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException atomicFailure) {
                Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        target.toFile().setReadable(true, false);
        target.toFile().setExecutable(true, false);
        return target;
    }

    static String bundledResourceName() {
        return "native/" + currentPlatformId() + "/" + LIBRARY_FILE_NAME;
    }

    static String currentPlatformId() {
        return platformId(System.getProperty("os.name", ""), System.getProperty("os.arch", ""));
    }

    static String platformId(String osName, String osArch) {
        return normalizeOs(osName) + "-" + normalizeArch(osArch);
    }

    private static Path cacheRoot() {
        String explicit = System.getProperty(CACHE_DIR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Path.of(explicit.trim());
        }
        return Path.of(System.getProperty("user.home"), ".synaptik", "native");
    }

    private static String normalizeOs(String osName) {
        String os = osName == null ? "" : osName.toLowerCase(Locale.ROOT);
        if (os.contains("mac")) {
            return "macos";
        }
        if (os.contains("linux")) {
            return "linux";
        }
        if (os.contains("windows")) {
            return "windows";
        }
        return sanitize(os);
    }

    private static String normalizeArch(String osArch) {
        String arch = osArch == null ? "" : osArch.toLowerCase(Locale.ROOT);
        if (arch.equals("aarch64") || arch.equals("arm64")) {
            return "arm64";
        }
        if (arch.equals("x86_64") || arch.equals("amd64")) {
            return "x64";
        }
        return sanitize(arch);
    }

    private static String sanitize(String value) {
        return value.replaceAll("[^a-z0-9_]+", "_");
    }

    private static String sha256(byte[] bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 digest is unavailable.", e);
        }
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? "<no-message>" : message;
    }
}
