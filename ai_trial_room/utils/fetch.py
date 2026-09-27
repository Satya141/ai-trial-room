"""Fetch a garment image from a URL, including a shop's product page.

A salesperson does not want to download a photo and re-upload it. They want to
paste the supplier's link. Two cases, handled transparently:

*Direct image URL* - ``https://shop.example/sarees/0421.jpg``
    Downloaded directly.

*Product page* - ``https://shop.example/products/kanjivaram-silk-saree``
    The HTML is fetched and the main product image is extracted from the page's
    Open Graph / Twitter Card metadata, which practically every e-commerce
    platform emits for link previews.

Security
--------
This accepts a URL from an untrusted user and fetches it server-side, which is a
textbook **SSRF** vector. On a public Space that request originates inside
someone else's infrastructure, so the guards here are not optional:

* only ``http`` and ``https``;
* every resolved IP is checked against loopback, private, link-local, reserved
  and multicast ranges - so ``localhost``, ``169.254.169.254`` (cloud metadata)
  and ``10.x`` internal services are unreachable;
* redirects are followed manually so **each hop is re-validated** - a public
  hostname that redirects to ``127.0.0.1`` is the standard bypass;
* responses are size-capped while streaming, not after, so a multi-gigabyte body
  cannot exhaust memory;
* a hard timeout, and a cap on redirect depth.

Nothing here trusts the URL, the DNS result, or the server's own headers.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Final

from PIL import Image

from ai_trial_room.utils.errors import InvalidInputError
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Allowed URL schemes.
ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

#: Maximum bytes accepted for an image.
MAX_IMAGE_BYTES: Final[int] = 20 * 1024 * 1024

#: Maximum bytes of HTML read while looking for a product image.
MAX_HTML_BYTES: Final[int] = 2 * 1024 * 1024

#: Per-request timeout in seconds.
TIMEOUT_S: Final[int] = 15

#: Maximum redirects followed. Each hop is re-validated.
MAX_REDIRECTS: Final[int] = 4

#: Sent so shops that block default Python clients still serve us.
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; AI-Trial-Room/1.0; +https://huggingface.co/spaces)"
)

#: Content types accepted as an image.
_IMAGE_TYPES: Final[tuple[str, ...]] = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/avif",
)

#: Metadata properties checked, in order of preference, for a product image.
_META_PATTERNS: Final[tuple[str, ...]] = (
    r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::secure_url)?["\']',
    r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)["\']',
    r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
)


@dataclass(frozen=True)
class FetchedImage:
    """An image successfully retrieved from a URL."""

    image: Image.Image
    source_url: str
    """The URL the bytes actually came from, after redirects and extraction."""

    from_product_page: bool
    """True when the image was extracted from an HTML page's metadata."""

    size_bytes: int

    def describe(self) -> str:
        """One-line summary for the UI."""
        origin = "product page" if self.from_product_page else "direct image"
        return (
            f"Loaded {self.image.width}x{self.image.height} "
            f"({self.size_bytes / 1024:.0f} KB) from {origin}"
        )


def _reject(message: str, detail: str) -> InvalidInputError:
    """Build a user-safe error, logging the technical reason."""
    logger.warning("URL fetch rejected: %s", detail)
    return InvalidInputError(message, detail=detail)


def _validate_url(url: str) -> urllib.parse.ParseResult:
    """Parse and sanity-check a URL before any network access.

    Raises
    ------
    InvalidInputError
        For a malformed URL or a disallowed scheme.
    """
    candidate = (url or "").strip()
    if not candidate:
        raise _reject("Please paste a garment image or product link.", "empty url")

    # Users paste bare domains, so a missing scheme becomes https. But that
    # upgrade must not swallow an opaque scheme: naively prepending would turn
    # ``data:image/png;base64,...`` into ``https://data:image/...``, whose
    # hostname is "data" and which would then pass the scheme check.
    scheme_match = re.match(r"^([A-Za-z][A-Za-z0-9+.\-]*):(//)?", candidate)
    if scheme_match is None:
        candidate = f"https://{candidate}"
    elif not scheme_match.group(2):
        scheme = scheme_match.group(1).lower()
        remainder = candidate[scheme_match.end() :]
        if scheme in ALLOWED_SCHEMES:
            pass  # "http:example.com" - unusual but let urlparse decide
        elif remainder.split("/")[0].isdigit():
            candidate = f"https://{candidate}"  # host:port, not a scheme
        else:
            raise _reject(
                "Only http and https links are supported.",
                f"opaque scheme {scheme!r}",
            )

    try:
        parsed = urllib.parse.urlparse(candidate)
    except ValueError as exc:
        raise _reject("That link could not be read.", f"unparseable: {exc}") from exc

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise _reject(
            "Only http and https links are supported.",
            f"scheme={parsed.scheme!r}",
        )
    if not parsed.hostname:
        raise _reject("That link has no website address in it.", "no hostname")

    return parsed


def _assert_public_host(hostname: str) -> None:
    """Resolve ``hostname`` and reject any non-public address.

    This is the SSRF guard. Every address the name resolves to is checked, since
    a hostname can return several and a single private one is enough to abuse.

    Raises
    ------
    InvalidInputError
        When the host does not resolve, or resolves to a non-public address.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise _reject(
            "That website could not be found. Check the link and try again.",
            f"dns failure for {hostname}: {exc}",
        ) from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise _reject("That website could not be found.", f"no addresses for {hostname}")

    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            raise _reject(
                "That link could not be verified.", f"unparseable address {raw!r}"
            ) from None

        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise _reject(
                "That link points to a private address, which is not allowed. "
                "Please use a public image or product link.",
                f"blocked address {address} for host {hostname}",
            )


def _open(url: str) -> tuple[Any, str]:
    """Open ``url``, following redirects manually and validating each hop.

    Returns
    -------
    tuple
        ``(response, final_url)``. The caller must close the response.

    Raises
    ------
    InvalidInputError
        On a blocked host, too many redirects, or a network failure.
    """
    current = url

    for hop in range(MAX_REDIRECTS + 1):
        parsed = _validate_url(current)
        assert parsed.hostname is not None
        _assert_public_host(parsed.hostname)

        request = urllib.request.Request(
            parsed.geturl(),
            headers={"User-Agent": USER_AGENT, "Accept": "image/*,text/html;q=0.8"},
            method="GET",
        )

        # Redirects are handled here rather than by urllib, so that a public
        # host redirecting to a private one is caught instead of followed.
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            response = opener.open(request, timeout=TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                location = exc.headers.get("Location")
                exc.close()
                if not location:
                    raise _reject(
                        "That link redirected somewhere unreadable.",
                        f"{exc.code} with no Location",
                    ) from None
                current = urllib.parse.urljoin(current, location)
                logger.debug("Redirect %d -> %s", hop + 1, current)
                continue
            raise _reject(
                f"The website returned an error ({exc.code}). "
                "Check the link, or download the image and upload it instead.",
                f"http {exc.code} for {current}",
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise _reject(
                "Could not reach that website. Check the link and your connection.",
                f"network error for {current}: {exc}",
            ) from exc

        return response, current

    raise _reject(
        "That link redirects too many times.", f"exceeded {MAX_REDIRECTS} redirects"
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn redirects into errors so :func:`_open` can re-validate each hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        """Never follow automatically."""
        return None


def _read_capped(response: Any, limit: int, *, what: str) -> bytes:
    """Read at most ``limit`` bytes, streaming so a huge body cannot exhaust memory.

    Raises
    ------
    InvalidInputError
        When the body exceeds ``limit``.
    """
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise _reject(
                f"That {what} is too large (over {limit // (1024 * 1024)} MB). "
                "Please use a smaller image.",
                f"{what} exceeded {limit} bytes",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _extract_image_url(html: str, base_url: str) -> str:
    """Find a product image URL in a page's link-preview metadata.

    Raises
    ------
    InvalidInputError
        When no candidate is present.
    """
    for pattern in _META_PATTERNS:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            found = urllib.parse.urljoin(base_url, match.group(1).strip())
            logger.info("Extracted product image from page metadata: %s", found)
            return found

    raise _reject(
        "No product image could be found on that page. Right-click the garment "
        "photo, copy the image address, and paste that instead.",
        "no og:image / twitter:image / image_src in html",
    )


def fetch_image(url: str, *, allow_product_page: bool = True) -> FetchedImage:
    """Fetch an image from a direct URL or a shop's product page.

    Parameters
    ----------
    url:
        A direct image URL, or an e-commerce product page.
    allow_product_page:
        When True, an HTML response is scanned for a product image. Set False to
        require a direct image URL.

    Returns
    -------
    FetchedImage

    Raises
    ------
    InvalidInputError
        For an invalid, blocked, oversized or unreadable target. Every message is
        safe to show the user and suggests what to do instead.
    """
    response, final_url = _open(url)

    try:
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()

        if content_type.startswith("text/html") or content_type in {
            "application/xhtml+xml"
        }:
            if not allow_product_page:
                raise _reject(
                    "That link is a web page, not an image.",
                    f"content-type={content_type}",
                )
            html = _read_capped(response, MAX_HTML_BYTES, what="page").decode(
                "utf-8", errors="replace"
            )
            response.close()

            image_url = _extract_image_url(html, final_url)
            nested = fetch_image(image_url, allow_product_page=False)
            return FetchedImage(
                image=nested.image,
                source_url=nested.source_url,
                from_product_page=True,
                size_bytes=nested.size_bytes,
            )

        # Some CDNs serve images as octet-stream; let Pillow be the arbiter
        # rather than trusting or over-trusting the header.
        if content_type and not (
            content_type in _IMAGE_TYPES or content_type.startswith("image/")
        ):
            if content_type != "application/octet-stream":
                raise _reject(
                    "That link is not an image. Paste a link to the garment photo, "
                    "or upload the file instead.",
                    f"content-type={content_type}",
                )

        payload = _read_capped(response, MAX_IMAGE_BYTES, what="image")
    finally:
        try:
            response.close()
        except Exception:  # noqa: BLE001 - already closed on the HTML path
            pass

    if not payload:
        raise _reject("That link returned an empty file.", "zero-length body")

    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types
        raise _reject(
            "That file could not be read as an image. It may be a video, a PDF, "
            "or a page that needs a login.",
            f"pillow failed: {type(exc).__name__}: {exc}",
        ) from exc

    logger.info(
        "Fetched %dx%d image (%.0f KB) from %s",
        image.width,
        image.height,
        len(payload) / 1024,
        final_url,
    )
    return FetchedImage(
        image=image.convert("RGB"),
        source_url=final_url,
        from_product_page=False,
        size_bytes=len(payload),
    )
