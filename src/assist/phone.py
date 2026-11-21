"""Open the reader on a phone: the laptop's Wi-Fi address and a QR code to scan.

The phone's own camera then takes the photo. Over plain http, phone browsers block the live camera
preview (it needs https), but the "take or upload a photo" button still opens the phone camera app.
"""
from __future__ import annotations

import socket

import cv2
import numpy as np


def lan_address() -> str | None:
    """This computer's address on the local network (e.g. 192.168.1.10), or None if offline.

    Connecting a UDP socket only selects the network interface; no packet is sent.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


def phone_url(port: int, https: bool = False) -> str | None:
    ip = lan_address()
    return f"{'https' if https else 'http'}://{ip}:{port}" if ip else None


def qr_image(text: str, size: int = 300) -> np.ndarray:
    """Black-on-white QR code (grayscale) with a quiet border, scaled up without blurring."""
    code = cv2.QRCodeEncoder.create().encode(text)
    code = cv2.resize(code, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(code, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
