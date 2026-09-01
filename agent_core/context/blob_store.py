"""Content-addressed filesystem blob storage."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from agent_core.context.models import BlobRef


class FileBlobStore:
    """Atomic, read-only blobs rooted outside agent-writable workspaces."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("invalid SHA-256 digest")
        return self.root / digest[:2] / digest[2:]

    async def put(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> BlobRef:
        digest = hashlib.sha256(data).hexdigest()
        await asyncio.to_thread(self._write, digest, data)
        return BlobRef(digest=digest, size=len(data), media_type=media_type)

    def _write(self, digest: str, data: bytes) -> None:
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=".blob-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o444)
            with suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    async def get(self, ref: BlobRef) -> bytes:
        data = await asyncio.to_thread(self._path(ref.digest).read_bytes)
        if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.digest:
            raise ValueError("blob integrity check failed")
        return data

    async def delete(self, digest: str) -> None:
        await asyncio.to_thread(self._path(digest).unlink, missing_ok=True)


__all__ = ["FileBlobStore"]
