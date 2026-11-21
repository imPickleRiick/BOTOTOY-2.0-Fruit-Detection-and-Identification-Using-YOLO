"""Self-signed HTTPS certificate so a phone on the same Wi-Fi can use the LIVE camera.

Phone browsers allow the camera only on secure (https) pages. `python app.py --https` calls
`ensure_certificate()`, which creates a certificate for this computer's current Wi-Fi address with
OpenSSL (included with Git for Windows), and serves it through src/assist/https_proxy.py. The phone
shows a one-time "not private" warning because the certificate is self-made; after accepting it, the
live camera works. For local demos only.

The key and certificate live in .streamlit/certs/ (git-ignored) and are re-created when the Wi-Fi
address changes.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.assist.phone import lan_address
from src.utils.helpers import PROJECT_ROOT

CERT_DIR = PROJECT_ROOT / ".streamlit" / "certs"
OPENSSL_CANDIDATES = [r"C:\Program Files\Git\usr\bin\openssl.exe", r"C:\Program Files\Git\mingw64\bin\openssl.exe"]


def find_openssl() -> str | None:
    found = shutil.which("openssl")
    if found:
        return found
    return next((p for p in OPENSSL_CANDIDATES if Path(p).is_file()), None)


def ensure_certificate() -> tuple[Path, Path, str | None]:
    """Return (cert, key, lan_ip), creating a new 30-day certificate if missing or for another address."""
    ip = lan_address()
    cert, key, stamp = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "address.txt"
    names = ["DNS:localhost", "IP:127.0.0.1"] + ([f"IP:{ip}"] if ip else [])
    wanted = ",".join(names)
    if cert.is_file() and key.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8") == wanted:
        return cert, key, ip
    openssl = find_openssl()
    if openssl is None:
        raise RuntimeError("OpenSSL was not found. Install Git for Windows (it includes OpenSSL) and try again.")
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "30",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=BOTOTOY Fruit Reader (local)",
                    "-addext", f"subjectAltName={wanted}"],
                   check=True, capture_output=True, timeout=60)
    stamp.write_text(wanted, encoding="utf-8")
    return cert, key, ip
