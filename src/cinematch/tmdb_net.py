"""Workaround for ISP DNS resolving the TMDB API to an unreachable edge IP.

On some networks ``api.themoviedb.org`` resolves to a dead address while the
CloudFront-backed IPs work fine. We pre-flight the connection and, when needed,
resolve via DNS-over-HTTPS and pin a reachable edge IP for this process by
wrapping ``socket.getaddrinfo``. Purely cosmetic on healthy networks (the
pre-flight finds the default resolution already works and nothing is patched).
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.parse
import urllib.request

_API_HOST = "api.themoviedb.org"
_ORIGINAL = socket.getaddrinfo
_patched = False
_patch_lock = threading.Lock()
_working_ips: list[str] = []


def _doh_hosts() -> list[str]:
    """A records for the API host via Google / Cloudflare DNS-over-HTTPS."""
    endpoints = [
        "https://dns.google/resolve?name={host}&type=A",
        "https://cloudflare-dns.com/dns-query?name={host}&type=A",
    ]
    for url in endpoints:
        try:
            req = urllib.request.Request(url.format(host=urllib.parse.quote(_API_HOST)))
            req.add_header("accept", "application/dns-json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode())
            ips = [a["data"] for a in payload.get("Answer", []) if a.get("type") == 1]
            if ips:
                return ips
        except Exception:
            continue
    return []


def _can_connect(ip: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def patch_tmdb_dns() -> bool:
    """Pin ``api.themoviedb.org`` to a reachable IP if the default is broken.

    Safe to call repeatedly; no-op when the normal resolution already works.
    Returns True when a working pin is installed (or one is not needed).
    """
    global _patched
    if _patched:
        return True
    with _patch_lock:
        if _patched:
            return True

        candidates: list[str] = []
        try:
            for entry in _ORIGINAL(_API_HOST, 443, socket.AF_UNSPEC, socket.SOCK_STREAM):
                ip = entry[4][0]
                if ip not in candidates:
                    candidates.append(ip)
        except OSError:
            pass
        for ip in _doh_hosts():
            if ip not in candidates:
                candidates.append(ip)

        working = [ip for ip in candidates if _can_connect(ip)]
        if not working:
            # Leave normal resolution in place; the caller will surface the error.
            _patched = True
            return False
        _working_ips[:] = working
        _patched = True

        def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            if host == _API_HOST and _working_ips:
                results = [
                    (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
                    for ip in _working_ips
                ]
                return results
            return _ORIGINAL(host, port, family, type, proto, flags)

        socket.getaddrinfo = patched_getaddrinfo
        return True
