from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

TECHNICAL_REJECTION = "回答草稿包含无法由当前证据或计算结果核验的事实，已停止返回该草稿。请重试。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    paths = {
        "service": root / "bankreg_trustrag" / "service.py",
        "verification": root / "bankreg_trustrag" / "verification.py",
        "retrieval": root / "bankreg_trustrag" / "retrieval_tools.py",
        "answer": root / "bankreg_trustrag" / "answer_generator.py",
    }

    texts = {}
    ok = True
    for key, path in paths.items():
        if not path.exists():
            print(f"{key}: MISSING FILE -> {path}")
            ok = False
            continue
        texts[key] = path.read_text(encoding="utf-8")
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"{key}: syntax OK")
        except Exception as exc:
            print(f"{key}: syntax FAILED -> {exc}")
            ok = False

    if len(texts) != len(paths):
        raise SystemExit(2)

    checks = {
        "选择题不再绕过 Agentic Planner": "and not has_choices" not in texts["service"],
        "Answer task refs 会展开为真实 evidence ids": "def _expanded_agentic_grounding_refs(" in texts["service"],
        "选择标签按选项事实证据核验": "def _grounded_choice_conclusion(" in texts["verification"],
        "source_hint 会解析/放宽真实入库标题": ("def _resolve_agentic_source_scope(" in texts["retrieval"] or "def _validated_source_filters(" in texts["retrieval"]),
        "Answer Agent 知道选项不是证据": "“选择A / A项正确”属于答案格式" in texts["answer"],
        "不再把内部草稿核验错误展示给用户": TECHNICAL_REJECTION not in texts["service"],
    }
    print()
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    print("\nagentic_full_fix=" + ("OK" if ok else "NOT_COMPLETE"))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
