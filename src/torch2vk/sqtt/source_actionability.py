"""Actionability heuristics for source-level profiler focus selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SourceActionability:
    score: int
    label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_source_actionability(text: str) -> SourceActionability:
    stripped = text.strip()
    if stripped == "" or stripped in {"{", "}"}:
        return SourceActionability(score=0, label="empty")
    if stripped.startswith("//"):
        return SourceActionability(score=0, label="comment")
    if stripped.startswith("#"):
        return SourceActionability(score=0, label="preprocessor")
    if stripped in {"break;", "continue;", "return;"}:
        return SourceActionability(score=1, label="control")
    if "barrier(" in stripped or "subgroup" in stripped or "coopMat" in stripped:
        return SourceActionability(score=5, label="synchronization-or-subgroup")
    if stripped.startswith(("for ", "for(", "while ", "while(")):
        return SourceActionability(score=5, label="loop")
    if any(token in stripped for token in ("shared_", "t_output[", "t_mat1[", "t_qweight_words[", "t_scales_and_zeros[")):
        return SourceActionability(score=5, label="device-memory")
    if any(token in stripped for token in ("gl_SubgroupInvocationID", "gl_WorkGroupID", "sizes.x", "sizes.y", "sizes.z", "sizes.w")):
        return SourceActionability(score=1, label="builtin-or-uniform-load")
    if "(" in stripped and ")" in stripped and not stripped.startswith(("if ", "if(", "switch ", "switch(")):
        if any(token in stripped for token in ("load_", "store_", "accumulate_", "fma(", "unpack_", "dot", "dequant")):
            return SourceActionability(score=5, label="compute-call")
        return SourceActionability(score=4, label="call")
    if stripped.startswith(("if ", "if(", "else", "switch ", "switch(", "case ", "return ")):
        return SourceActionability(score=2, label="control")
    if "[" in stripped or "]" in stripped:
        return SourceActionability(score=4, label="indexed-access")
    if "=" in stripped:
        if stripped.startswith("const ") or stripped.startswith(("uint ", "int ", "float ", "bool ")):
            return SourceActionability(score=2, label="declaration")
        return SourceActionability(score=3, label="assignment")
    return SourceActionability(score=2, label="expression")


def classify_hot_line_actionability(*, source_kind: str, source_text: str) -> SourceActionability:
    if source_kind != "glsl":
        return SourceActionability(score=0, label=source_kind)
    return classify_source_actionability(source_text)
