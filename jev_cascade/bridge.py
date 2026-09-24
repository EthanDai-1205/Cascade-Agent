"""The JSON-line bridge client that every tool shares.

The heavy lifting of each tool lives outside this package: the browser drives
Playwright from a Node process, and the desktop reads the macOS Accessibility
tree from a Swift one. Both helpers speak the same one-JSON-object-per-line
protocol over stdin and stdout, so one small client serves both, and the
Python package keeps its zero-dependencies property. One long-lived process
per tool, because relaunching it every step would lose state (form contents
in the browser, the desktop is the desktop).
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any


class BridgeError(RuntimeError):
    """A bridge process could not start or spoke something unexpected."""


class LineBridge:
    """One helper process; one JSON command in, one JSON reply out.

    ``what`` names the bridge in error messages ("browser bridge", "computer
    bridge"), because a dead helper should be diagnosed by its name, not by
    its process number.
    """

    def __init__(
        self,
        argv: list[str],
        what: str = "bridge",
        env_extra: dict[str, str] | None = None,
    ) -> None:
        self.what = what
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, **env_extra} if env_extra else None,
        )

    def exchange(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._proc.poll() is not None:
            raise BridgeError(
                f"the {self.what} exited with code {self._proc.returncode}: "
                f"{(self._proc.stderr.read() or '')[-300:].strip()}"
            )
        assert self._proc.stdin and self._proc.stdout
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise BridgeError(f"the {self.what} stopped listening: {exc}") from exc
        if not line:
            raise BridgeError(
                f"the {self.what} closed the connection: "
                f"{(self._proc.stderr.read() or '')[-300:].strip()}"
            )
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeError(
                f"the {self.what} said something that is not JSON: {line[:200]!r}"
            ) from exc
        if not reply.get("ok"):
            raise BridgeError(str(reply.get("error") or reply))
        return reply

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self.exchange({"cmd": "close"})
            except BridgeError:
                pass
            self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a wedged helper
            self._proc.kill()
            self._proc.wait(timeout=10)
        # Close the pipes too, or every session leaks three file descriptors.
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
