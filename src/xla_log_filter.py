"""
Hide XLA's multi-device "may be stuck" / "Thread is unstuck" messages.

With several CPU devices (pmap) XLA prints, whenever one device reaches a
collective (e.g. the sum over windows) more than 5 s after the others,

    ... This thread has been waiting for 5000ms ... and may be stuck ...
    ... Thread is unstuck! Warning above was a false-positive ...

They are harmless (devices just finish at different times) but flood the
console. Depending on the jaxlib version they are written by TensorFlow-style
logging (hidden by TF_CPP_MIN_LOG_LEVEL=3) or by absl logging (not affected by
it, and there is no XLA option for the 5 s threshold on CPU). So `install()`
filters them at the file-descriptor level: fd 2 (stderr) is redirected into a
pipe, a background thread forwards every line except these two messages to
the original stderr, and at exit the stream is restored and the number of
hidden lines is reported. Python tracebacks and all other messages pass
through unchanged.
"""
import atexit
import os
import re
import sys
import threading

_PATTERN = re.compile(rb"may be stuck|Thread is unstuck")
_state = {}


def install(pattern=_PATTERN):
    """Start filtering fd 2 (idempotent). Call before importing jax."""
    if _state:
        return
    try:
        sys.stderr.flush()
        orig = os.dup(2)
        r, w = os.pipe()
        os.dup2(w, 2)
        os.close(w)
    except OSError:
        return   # e.g. no real stderr: leave everything as it is
    hidden = [0]

    def pump():
        buf = b""
        while True:
            try:
                chunk = os.read(r, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            out = []
            for line in lines:
                if pattern.search(line):
                    hidden[0] += 1
                else:
                    out.append(line + b"\n")
            if out:
                os.write(orig, b"".join(out))
        if buf and not pattern.search(buf):
            os.write(orig, buf)
        os.close(r)

    t = threading.Thread(target=pump, name="xla-log-filter", daemon=True)
    t.start()
    _state.update(orig=orig, thread=t, hidden=hidden)

    def restore():
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(orig, 2)          # closes the pipe's last write end -> reader sees EOF
        t.join(timeout=5)
        if hidden[0]:
            os.write(orig, f"(hid {hidden[0]} harmless XLA 'may be stuck'/'unstuck' message line(s); "
                           "--xla-warnings shows them)\n".encode())

    atexit.register(restore)
