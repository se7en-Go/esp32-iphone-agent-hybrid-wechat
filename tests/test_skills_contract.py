"""Contract tests for skills/*/SKILL.md compatibility with the base server parser."""

from pathlib import Path

import pytest

from wechat_agent.config import repo_root
from wechat_agent.guards.skill_guard import parse_skill_frontmatter, validate_skills_dir


def test_repo_skills_valid_and_no_static_layouts() -> None:
    skills_dir = repo_root() / "skills"
    validated = validate_skills_dir(skills_dir)
    assert "wechat-send" in validated
    assert "wechat-group-reply" in validated
    assert list(skills_dir.glob("*/layout.json")) == []


def test_wechat_send_frontmatter_parses_like_base_server() -> None:
    md = (repo_root() / "skills" / "wechat-send" / "SKILL.md").read_text(encoding="utf-8")
    meta = parse_skill_frontmatter(md)
    assert meta["min_interval_seconds"] >= 3.0
    assert 1 <= meta["max_per_hour"] <= 30
    assert meta["quiet_hours"] == "23:00-07:00"
    assert "账号异常" in meta["extra_stop"]
    assert "功能受限" in meta["extra_stop"]


def test_rejects_yaml_list_in_extra_stop() -> None:
    bad_md = """---
min_interval_seconds: 10
extra_stop:
  - 账号异常
---
# body
"""
    with pytest.raises(ValueError, match="does not support YAML lists"):
        parse_skill_frontmatter(bad_md)


def test_rejects_bracket_list_in_extra_stop() -> None:
    bad_md = """---
extra_stop: ["账号异常", "风控"]
---
# body
"""
    with pytest.raises(ValueError, match="bracket lists"):
        parse_skill_frontmatter(bad_md)


def test_rejects_static_layout_json(tmp_path: Path) -> None:
    (tmp_path / "wechat-send").mkdir()
    (tmp_path / "wechat-send" / "SKILL.md").write_text("---\nmin_interval_seconds: 10\n---\n# ok\n", encoding="utf-8")
    (tmp_path / "wechat-send" / "layout.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Static layout.json is forbidden"):
        validate_skills_dir(tmp_path)
