"""A small HTTPS front door for the app, so a phone can use its LIVE camera.

Phone browsers allow the camera only on https pages. Streamlit's own https mode proved unreliable on
Windows in our tests (the live camera never uploaded and some files failed to load), so the app keeps
running on plain http and this proxy accepts TLS connections on another port and passes the decrypted
bytes, unchanged, to the app on 127.0.0.1. Because it copies raw bytes, normal pages, file uploads and
Streamlit's websocket all work.

For local demos on a private network only: the certificate is self-made (see https_cert.py).
"""
from __future__ import annotations

import select
import socket
import socketserver
import ssl
import threading
from pathlib import Path

IDLE_SECONDS = 600  # close connections that have been silent for 10 minutes


def _pipe(client: ssl.SSLSocket, upstream: socket.socket) -> None:
    """Copy bytes both ways until either side closes."""
    socks = [client, upstream]
    while True:
        readable = [client] if client.pending() else select.select(socks, [], [], IDLE_SECONDS)[0]
        if not readable:
            return
        for s in readable:
            data = s.recv(65536)
            if not data:
                return
            (upstream if s is client else client).sendall(data)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_https_proxy(cert: Path, key: Path, listen_port: int, target_port: int,
                      listen_host: str = "0.0.0.0") -> socketserver.BaseServer:
    """Start the proxy in a background thread and return the server (call .shutdown() to stop)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert), str(key))

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                client = context.wrap_socket(self.request, server_side=True)
            except (ssl.SSLError, OSError):
                return  # e.g. the phone rejected the certificate before accepting the warning
            try:
                with socket.create_connection(("127.0.0.1", target_port), timeout=10) as upstream:
                    upstream.settimeout(None)
                    _pipe(client, upstream)
            except OSError:
                pass
            finally:
                try:
                    client.close()
                except OSError:
                    pass

    server = _Server((listen_host, listen_port), Handler)
    threading.Thread(target=server.serve_forever, name="https-proxy", daemon=True).start()
    return server
