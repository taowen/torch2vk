"""Structured parser for compiler-native AMD disassembly bundles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_BLOCK_LABEL_PATTERN = re.compile(r"^(?P<label>[A-Za-z_][A-Za-z0-9_]*):\s*$")
_HEX_COMMENT_PATTERN = re.compile(r";\s*([0-9a-fA-F]{8}(?:\s+[0-9a-fA-F]{8})*)\s*$")


@dataclass(frozen=True, slots=True)
class DisasmInstruction:
    offset: int
    size_bytes: int
    text: str
    opcode: str
    block_label: str | None

    @property
    def end_offset(self) -> int:
        return self.offset + self.size_bytes


@dataclass(frozen=True, slots=True)
class CompilerNativeDisasmIndex:
    path: str
    instructions: tuple[DisasmInstruction, ...]
    block_offsets: dict[str, int]


def parse_compiler_native_disasm(path: Path) -> CompilerNativeDisasmIndex:
    text = path.read_text(encoding="utf-8", errors="replace")
    instructions: list[DisasmInstruction] = []
    block_offsets: dict[str, int] = {}
    current_block_label: str | None = None
    current_offset = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith(";"):
            continue
        block_match = _BLOCK_LABEL_PATTERN.match(stripped)
        if block_match is not None:
            label = block_match.group("label")
            current_block_label = label
            block_offsets[label] = current_offset
            continue
        hex_match = _HEX_COMMENT_PATTERN.search(line)
        if hex_match is None:
            continue
        encoding_words = hex_match.group(1).split()
        size_bytes = len(encoding_words) * 4
        instruction_text = line[: hex_match.start()].strip()
        opcode = instruction_text.split(maxsplit=1)[0] if instruction_text != "" else ""
        instructions.append(
            DisasmInstruction(
                offset=current_offset,
                size_bytes=size_bytes,
                text=instruction_text,
                opcode=opcode,
                block_label=current_block_label,
            )
        )
        current_offset += size_bytes
    return CompilerNativeDisasmIndex(
        path=str(path),
        instructions=tuple(instructions),
        block_offsets=block_offsets,
    )
