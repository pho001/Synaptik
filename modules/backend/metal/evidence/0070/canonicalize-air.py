#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def canonicalize_ir(text: str, work: Path, sdk: Path, root: Path) -> str:
    replacements = (
        (str(work.resolve()), "<WORK>"),
        (str(sdk.resolve()), "<SDKROOT>"),
        (str(root.resolve()), "<ROOT>"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(r"^.*(?:generated-fixtures|forbidden-floating-fixtures)\.air:\s+file format LLVM IR$", "<AIR>:\tfile format LLVM IR", text, flags=re.MULTILINE)
    text = re.sub(r'^source_filename = ".*"$', 'source_filename = "<GENERATED_SOURCE>"', text, flags=re.MULTILINE)
    text = re.sub(r"^; ModuleID = ['\"].*['\"]$", '; ModuleID = "<GENERATED_AIR>"', text, flags=re.MULTILINE)
    return text.replace("\r\n", "\n")


def canonicalize_nm(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").splitlines():
        lines.append(re.sub(r"^[0-9a-fA-F]+(?=\s)", "<ADDR>", line))
    return "\n".join(lines) + "\n"


def main() -> None:
    assert len(sys.argv) >= 6, (
        "usage: canonicalize-air.py WORK SDK ROOT AIR_LL AIR_NM [EXTRA_AIR_LL ...]"
    )
    work, sdk, root, ir_path, nm_path = map(Path, sys.argv[1:6])
    ir = canonicalize_ir(ir_path.read_text(encoding="utf-8"), work, sdk, root)
    nm = canonicalize_nm(nm_path.read_text(encoding="utf-8"))
    ir_path.with_suffix(ir_path.suffix + ".canonical").write_text(
        ir, encoding="utf-8", newline="\n"
    )
    nm_path.with_suffix(nm_path.suffix + ".canonical").write_text(
        nm, encoding="utf-8", newline="\n"
    )
    for extra in map(Path, sys.argv[6:]):
        canonical = canonicalize_ir(
            extra.read_text(encoding="utf-8"), work, sdk, root
        )
        extra.with_suffix(extra.suffix + ".canonical").write_text(
            canonical, encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()
