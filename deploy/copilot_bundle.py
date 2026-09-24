"""Build, verify, and stage a local 1PAVI Copilot transfer release.

Only Python's standard library is required. A bundle is never uploaded by this
tool, installed over an existing release, or activated as a running service.
"""

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
INDEX_FILES = (
    "data/bm25_index.pkl",
    "data/parent_store.pkl",
    "data/index_manifest.json",
)
CODE_REQUIRED = (
    "app/copilot_api.py",
    "app/copilot_auth.py",
    "app/copilot_chat_log.py",
    "app/copilot_feedback.py",
    "app/copilot_context.py",
    "app/copilot_settings.py",
    "app/rag_pipeline.py",
    "app/requirements-api.txt",
    "deploy/v5_ui_feedback.git.patch",
    "deploy/v5_podman_copilot.git.patch",
)
PRIVATE_REQUIRED = (*INDEX_FILES, "app/chroma_db/chroma.sqlite3")
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
RUNTIME_PACKAGES = (
    "ollama", "langchain", "langchain-community", "langchain-core",
    "langchain-text-splitters", "chromadb", "sentence-transformers",
    "transformers", "torch", "numpy", "rank-bm25", "fastapi",
    "uvicorn", "python-jose",
)


def _runtime_versions():
    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name):
    path = PurePosixPath(name)
    if not name or name in {".", ".."} or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe bundle member: {name!r}")
    if str(path) != name or "\\" in name:
        raise ValueError(f"Unsafe bundle member: {name!r}")
    return path


def _read_stable(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular source file: {path}")
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while packaging: {path}")
    return content


def _check_index_sources(files):
    for name in INDEX_FILES:
        if name not in files:
            raise ValueError(f"Missing index artifact: {name}")
    manifest = json.loads(files["data/index_manifest.json"])
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected index schema version 2")
    if not isinstance(manifest.get("expected_parent_count"), int) or manifest["expected_parent_count"] <= 0:
        raise ValueError("Index parent count is invalid")
    if manifest.get("indexed_parent_count") != manifest["expected_parent_count"]:
        raise ValueError("Index parent count is incomplete")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Index sources are missing")
    for source in sources:
        source_file = source.get("source_file")
        if not isinstance(source_file, str) or Path(source_file).name != source_file:
            raise ValueError("Unsafe index source name")
        name = f"data/{source_file}"
        if name not in files or _digest(files[name]) != source.get("sha256"):
            raise ValueError(f"Manual/index hash mismatch: {source_file}")
    return manifest


def _sqlite_quick_check(content, expected_child_count=None):
    with tempfile.TemporaryDirectory(prefix="copilot-sqlite-check-") as scratch:
        database = Path(scratch) / "chroma.sqlite3"
        database.write_bytes(content)
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        try:
            result = connection.execute("PRAGMA quick_check").fetchone()
            if expected_child_count is not None:
                indexed_children = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        finally:
            connection.close()
        if result != ("ok",):
            raise ValueError("Chroma SQLite snapshot failed quick_check")
        if expected_child_count is not None and indexed_children != expected_child_count:
            raise ValueError("Chroma child count differs from index manifest")


def _sqlite_snapshot(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Chroma SQLite database is missing")
    with tempfile.TemporaryDirectory(prefix="copilot-sqlite-backup-") as scratch:
        target_path = Path(scratch) / "chroma.sqlite3"
        source = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        target = sqlite3.connect(target_path)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        content = target_path.read_bytes()
    _sqlite_quick_check(content)
    return content


def _collect_code(project_root):
    selected = set(project_root.joinpath("app").glob("*.py"))
    selected.update(project_root.joinpath("app/rag_core").glob("*.py"))
    selected.update(project_root.joinpath("app/pages").glob("*.py"))
    selected.update(project_root / name for name in (
        "app/requirements-api.txt",
        "deploy/copilot_bundle.py",
        "deploy/README.md",
        "deploy/v5_ui_feedback.git.patch",
        "deploy/v5_podman_copilot.git.patch",
        "docs/README.md",
    ))
    files = {
        path.relative_to(project_root).as_posix(): _read_stable(path)
        for path in sorted(selected)
    }
    if any(name not in files for name in CODE_REQUIRED):
        raise ValueError("Copilot code package is incomplete")
    return files


def _collect_private(project_root):
    index_bytes = _read_stable(project_root / "data/index_manifest.json")
    index_manifest = json.loads(index_bytes)
    source_names = set()
    for source in index_manifest.get("sources", []):
        source_file = source.get("source_file")
        if not isinstance(source_file, str) or Path(source_file).name != source_file:
            raise ValueError("Unsafe index source name")
        source_names.add(f"data/{source_file}")
    selected = set(INDEX_FILES) | source_names
    files = {name: _read_stable(project_root / name) for name in sorted(selected)}
    manifest = _check_index_sources(files)
    chroma_dir = project_root / "app/chroma_db"
    if not chroma_dir.is_dir() or chroma_dir.is_symlink():
        raise ValueError("Chroma index directory is missing")
    files["app/chroma_db/chroma.sqlite3"] = _sqlite_snapshot(chroma_dir / "chroma.sqlite3")
    _sqlite_quick_check(files["app/chroma_db/chroma.sqlite3"], manifest.get("indexed_child_count"))
    for path in sorted(chroma_dir.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"Symlink in Chroma index: {path}")
            continue
        if path.name in {"chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"}:
            continue
        name = path.relative_to(project_root).as_posix()
        _safe_name(name)
        files[name] = _read_stable(path)
    if len(files) <= len(INDEX_FILES) + len(source_names) + 1:
        raise ValueError("Chroma vector index files are missing")
    return files


def _write_archive(path, kind, release_id, files):
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "release_id": release_id,
        "files": {name: _digest(content) for name, content in sorted(files.items())},
    }
    payload = {"bundle-manifest.json": json.dumps(manifest, sort_keys=True).encode("utf-8")}
    payload.update(files)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for name, content in sorted(payload.items()):
            _safe_name(name)
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o600 if kind == "private_data" else 0o644
            entry.mtime = 0
            archive.addfile(entry, io.BytesIO(content))


def build_bundle(project_root=PROJECT_ROOT, output_root=None, release_id=None):
    project_root = Path(project_root).resolve()
    output_root = Path(output_root or project_root / "deploy/out").resolve()
    code = _collect_code(project_root)
    private = _collect_private(project_root)
    content_digest = _digest(
        json.dumps({
            "code": {name: _digest(content) for name, content in code.items()},
            "private": {name: _digest(content) for name, content in private.items()},
        }, sort_keys=True).encode("utf-8")
    )[:8]
    release_id = release_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + content_digest
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError("Release ID contains unsafe characters")
    output_root.mkdir(parents=True, exist_ok=True)
    release_dir = output_root / release_id
    release_dir.mkdir(mode=0o700, exist_ok=False)
    code_path = release_dir / "code.tar.gz"
    private_path = release_dir / "private-data.tar.gz"
    _write_archive(code_path, "code", release_id, code)
    _write_archive(private_path, "private_data", release_id, private)
    envelope = {
        "format_version": FORMAT_VERSION,
        "release_id": release_id,
        "archives": {
            code_path.name: _file_digest(code_path),
            private_path.name: _file_digest(private_path),
        },
        "index_schema_version": 2,
        "embedding_model": json.loads(private["data/index_manifest.json"])["embedding_model"],
        "runtime_reference": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "packages": _runtime_versions(),
        },
    }
    (release_dir / "bundle.json").write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(project_root / "deploy/copilot_bundle.py", release_dir / "copilot_bundle.py")
    shutil.copy2(project_root / "deploy/README.md", release_dir / "README.md")
    shutil.copy2(project_root / "deploy/v5_ui_feedback.git.patch", release_dir / "v5_ui_feedback.git.patch")
    shutil.copy2(project_root / "deploy/v5_podman_copilot.git.patch", release_dir / "v5_podman_copilot.git.patch")
    return release_dir


def _read_archive(path, expected_kind, release_id):
    files = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            _safe_name(member.name)
            if not member.isfile() or member.name in files:
                raise ValueError("Bundle contains a link, directory, or duplicate member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Bundle member cannot be read")
            files[member.name] = handle.read()
    if "bundle-manifest.json" not in files:
        raise ValueError("Bundle manifest is missing")
    manifest = json.loads(files.pop("bundle-manifest.json"))
    if (manifest.get("format_version"), manifest.get("kind"), manifest.get("release_id")) != (
        FORMAT_VERSION, expected_kind, release_id
    ):
        raise ValueError("Bundle type or release ID does not match")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or set(expected) != set(files):
        raise ValueError("Bundle file list does not match its manifest")
    for name, content in files.items():
        if _digest(content) != expected[name]:
            raise ValueError(f"Bundle file hash mismatch: {name}")
    return files


def verify_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir).resolve()
    envelope = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    release_id = envelope.get("release_id")
    if envelope.get("format_version") != FORMAT_VERSION or not isinstance(release_id, str) or not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError("Unsupported or unsafe bundle envelope")
    expected_archives = envelope.get("archives")
    if not isinstance(expected_archives, dict) or set(expected_archives) != {"code.tar.gz", "private-data.tar.gz"}:
        raise ValueError("Bundle archive list is invalid")
    for name, digest in expected_archives.items():
        if _file_digest(bundle_dir / name) != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")
    code = _read_archive(bundle_dir / "code.tar.gz", "code", release_id)
    private = _read_archive(bundle_dir / "private-data.tar.gz", "private_data", release_id)
    if any(name not in code for name in CODE_REQUIRED) or any(name not in private for name in PRIVATE_REQUIRED):
        raise ValueError("Release is missing a required Copilot component")
    index_manifest = _check_index_sources(private)
    _sqlite_quick_check(private["app/chroma_db/chroma.sqlite3"], index_manifest.get("indexed_child_count"))
    if envelope.get("embedding_model") != index_manifest.get("embedding_model"):
        raise ValueError("Embedding model differs from index manifest")
    return envelope, code, private


def install_bundle(bundle_dir, install_root):
    envelope, code, private = verify_bundle(bundle_dir)
    install_root = Path(install_root).expanduser().resolve()
    if not install_root.is_absolute() or len(install_root.parts) < 4:
        raise ValueError("Choose a dedicated absolute install root")
    releases = install_root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    release_dir = releases / envelope["release_id"]
    if release_dir.exists():
        raise FileExistsError(f"Release already exists; refusing to overwrite: {release_dir}")
    staged = Path(tempfile.mkdtemp(prefix=f".{envelope['release_id']}-", dir=releases))
    try:
        for kind, files in (("code", code), ("private_data", private)):
            for name, content in files.items():
                relative = _safe_name(name)
                destination = staged.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with destination.open("xb") as handle:
                    handle.write(content)
                destination.chmod(0o600 if kind == "private_data" else 0o644)
        (staged / "bundle.json").write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        os.rename(staged, release_dir)
    except Exception:
        # Only the new, mkdtemp-created staging tree is removed on failure.
        shutil.rmtree(staged)
        raise
    return release_dir


def preflight_bundle(bundle_dir, jwt_public_key_path):
    """Report missing target prerequisites without loading GPU models."""
    envelope, _, _ = verify_bundle(bundle_dir)
    versions = _runtime_versions()
    cache_root = Path(
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or Path.home() / ".cache/huggingface/hub"
    )
    snapshots = cache_root / "models--BAAI--bge-reranker-v2-m3/snapshots"
    reranker_cached = snapshots.is_dir() and any(item.is_dir() for item in snapshots.iterdir())
    ollama_names = set()
    ollama_reachable = False
    try:
        with urllib_request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            tags = json.load(response)
        ollama_names = {item.get("name") or item.get("model") for item in tags.get("models", [])}
        ollama_reachable = True
    except (OSError, ValueError, urllib_error.URLError):
        pass
    required_models = {"gpt-oss:20b", envelope["embedding_model"]}
    public_key_present = Path(jwt_public_key_path).expanduser().is_file()
    missing_packages = sorted(name for name, version in versions.items() if version is None)
    return {
        "release_id": envelope["release_id"],
        "python": platform.python_version(),
        "machine": platform.machine(),
        "source_runtime_reference": envelope.get("runtime_reference"),
        "missing_packages": missing_packages,
        "ollama_reachable": ollama_reachable,
        "missing_ollama_models": sorted(required_models - ollama_names),
        "bge_reranker_cached": reranker_cached,
        "jwt_public_key_present": public_key_present,
        "ready_for_api_start": (
            not missing_packages and ollama_reachable
            and required_models.issubset(ollama_names)
            and reranker_cached and public_key_present
        ),
    }


def render_service(release_dir, python_path, jwt_public_key_path, user=None):
    release_dir = Path(release_dir).expanduser().resolve()
    # Keep the venv launcher path: resolving its symlink would bypass the
    # environment and start system Python without the installed RAG packages.
    python_path = Path(python_path).expanduser().absolute()
    jwt_public_key_path = Path(jwt_public_key_path).expanduser().resolve()
    user = user or getpass.getuser()
    if not release_dir.is_dir() or not (release_dir / "app/copilot_api.py").is_file():
        raise ValueError("A staged Copilot release is required")
    if not python_path.is_file() or not jwt_public_key_path.is_file():
        raise ValueError("Target Python or 1PAVI JWT public key is missing")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user):
        raise ValueError("Unsafe systemd user name")
    if any(any(character.isspace() or character in {'"', "'"} for character in str(path)) for path in (
        release_dir, python_path, jwt_public_key_path
    )):
        raise ValueError("Service paths cannot contain whitespace or quotes")
    service = f"""[Unit]
Description=1PAVI local Copilot RAG API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={release_dir}
Environment=COPILOT_AUTH_REQUIRED=true
Environment=COPILOT_WARMUP=true
Environment=COPILOT_RUNTIME_PROFILE=production_bge
Environment=COPILOT_MODEL=gpt-oss:20b
Environment=COPILOT_PAGE_CONTEXT_ENABLED=false
Environment=COPILOT_CLARIFICATION_ENABLED=false
Environment=COPILOT_FEEDBACK_LOG_PATH={release_dir.parent.parent / 'log/user_feedback_logs.jsonl'}
Environment=COPILOT_CHAT_LOG_PATH={release_dir.parent.parent / 'log/chat_logs.log'}
Environment=JWT_PUBLIC_KEY_PATH={jwt_public_key_path}
ExecStart={python_path} -m uvicorn app.copilot_api:app --host 0.0.0.0 --port 49159 --workers 1
Restart=on-failure
RestartSec=5
TimeoutStartSec=180
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
"""
    output = release_dir / "1pavi-copilot.service"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(service)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="Create code and private-data archives locally")
    build.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    build.add_argument("--output-root", type=Path)
    build.add_argument("--release-id")
    verify = actions.add_parser("verify", help="Check archives, manuals, and Chroma snapshot")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    install = actions.add_parser("install", help="Stage a new release without activating it")
    install.add_argument("--bundle-dir", type=Path, required=True)
    install.add_argument("--install-root", type=Path, required=True)
    preflight = actions.add_parser("preflight", help="Check target runtime without GPU/model loading")
    preflight.add_argument("--bundle-dir", type=Path, required=True)
    preflight.add_argument("--jwt-public-key", type=Path, required=True)
    service = actions.add_parser("service", help="Render a path-specific unit without enabling it")
    service.add_argument("--release-dir", type=Path, required=True)
    service.add_argument("--python", type=Path, required=True)
    service.add_argument("--jwt-public-key", type=Path, required=True)
    service.add_argument("--user")
    args = parser.parse_args(argv)
    try:
        if args.action == "build":
            print(build_bundle(args.project_root, args.output_root, args.release_id))
        elif args.action == "verify":
            envelope, code, private = verify_bundle(args.bundle_dir)
            print(f"verified {envelope['release_id']}: {len(code)} code files, {len(private)} private files, index schema 2")
        elif args.action == "install":
            print(install_bundle(args.bundle_dir, args.install_root))
        elif args.action == "preflight":
            report = preflight_bundle(args.bundle_dir, args.jwt_public_key)
            print(json.dumps(report, indent=2))
            return 0 if report["ready_for_api_start"] else 2
        else:
            print(render_service(args.release_dir, args.python, args.jwt_public_key, args.user))
    except (OSError, ValueError, RuntimeError, sqlite3.Error, tarfile.TarError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
