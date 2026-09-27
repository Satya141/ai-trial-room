"""Tests for fetching a garment from a URL.

The SSRF guards matter most. This feature takes a URL from an untrusted user and
fetches it server-side, and on a public Space that request originates inside
someone else's infrastructure — so ``localhost``, private ranges and cloud
metadata endpoints must be unreachable, including via a redirect from a public
host.

Network-dependent tests are skipped when offline rather than failing, so the
suite stays runnable on a plane. The guard tests need no network at all, because
they reject before any request is made.

Run standalone::

    python tests/test_fetch.py
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.utils.errors import InvalidInputError  # noqa: E402
from ai_trial_room.utils.fetch import (  # noqa: E402
    ALLOWED_SCHEMES,
    MAX_HTML_BYTES,
    MAX_IMAGE_BYTES,
    MAX_REDIRECTS,
    TIMEOUT_S,
    FetchedImage,
    _assert_public_host,
    _extract_image_url,
    _validate_url,
    fetch_image,
)

#: A stable public image and a page carrying og:image, used for the live tests.
LIVE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/4/44/Sari_on_mannequin_for_demo.jpg"
)
LIVE_PAGE_URL = "https://en.wikipedia.org/wiki/Sari"


def _online(host: str = "upload.wikimedia.org") -> bool:
    """True when DNS resolves, used to skip network tests offline."""
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# SSRF guards — no network needed, these reject before any request
# --------------------------------------------------------------------------- #


def test_loopback_addresses_are_blocked() -> None:
    """localhost must be unreachable from a user-supplied URL."""
    for url in (
        "http://localhost/x.jpg",
        "http://127.0.0.1/x.jpg",
        "http://127.0.0.1:8080/admin",
        "http://[::1]/x.jpg",
    ):
        parsed = _validate_url(url)
        assert parsed.hostname
        try:
            _assert_public_host(parsed.hostname)
        except InvalidInputError as exc:
            assert "private address" in exc.user_message
        else:  # pragma: no cover
            raise AssertionError(f"not blocked: {url}")


def test_cloud_metadata_endpoint_is_blocked() -> None:
    """169.254.169.254 leaks cloud credentials; it must never be fetched."""
    parsed = _validate_url("http://169.254.169.254/latest/meta-data/iam/")
    assert parsed.hostname
    try:
        _assert_public_host(parsed.hostname)
    except InvalidInputError as exc:
        assert "private address" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("cloud metadata endpoint was not blocked")


def test_private_ranges_are_blocked() -> None:
    """RFC1918 and friends must be unreachable."""
    for host in ("10.0.0.5", "192.168.1.1", "172.16.0.1", "0.0.0.0"):
        try:
            _assert_public_host(host)
        except InvalidInputError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"not blocked: {host}")


def test_non_http_schemes_are_rejected() -> None:
    """file:// would read the server's disk; only http(s) is allowed."""
    assert ALLOWED_SCHEMES == frozenset({"http", "https"})

    for url in ("file:///etc/passwd", "ftp://host/x.jpg", "gopher://host", "data:image/png;base64,AA"):
        try:
            _validate_url(url)
        except InvalidInputError as exc:
            assert "http" in exc.user_message.lower()
        else:  # pragma: no cover
            raise AssertionError(f"not blocked: {url}")


def test_empty_url_is_rejected_with_guidance() -> None:
    """An empty paste must say what to do, not raise something opaque."""
    try:
        _validate_url("   ")
    except InvalidInputError as exc:
        assert "paste" in exc.user_message.lower()
    else:  # pragma: no cover
        raise AssertionError("empty URL accepted")


def test_bare_domain_is_upgraded_to_https() -> None:
    """Users paste domains without a scheme; assume https rather than failing."""
    assert _validate_url("shop.example/a.jpg").scheme == "https"
    assert _validate_url("http://shop.example/a.jpg").scheme == "http"


def test_url_without_a_host_is_rejected() -> None:
    """A scheme with no hostname cannot be fetched."""
    try:
        _validate_url("https:///path/only")
    except InvalidInputError as exc:
        assert "website address" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("hostless URL accepted")


def test_unresolvable_host_gives_a_friendly_message() -> None:
    """A typo'd domain must read like a typo, not a stack trace."""
    try:
        _assert_public_host("this-domain-should-not-exist-aitr-test.invalid")
    except InvalidInputError as exc:
        assert "could not be found" in exc.user_message
        assert "dns" in (exc.detail or "").lower()
    else:  # pragma: no cover
        raise AssertionError("unresolvable host accepted")


def test_limits_are_sane() -> None:
    """The caps must exist and be small enough to matter."""
    assert 0 < MAX_IMAGE_BYTES <= 50 * 1024 * 1024
    assert 0 < MAX_HTML_BYTES < MAX_IMAGE_BYTES
    assert 0 < TIMEOUT_S <= 30
    assert 0 < MAX_REDIRECTS <= 10


# --------------------------------------------------------------------------- #
# Product-page extraction — pure parsing, no network
# --------------------------------------------------------------------------- #


def test_extracts_og_image() -> None:
    """Open Graph is what practically every shop emits for link previews."""
    html = '<html><head><meta property="og:image" content="/img/saree.jpg"></head></html>'
    assert _extract_image_url(html, "https://shop.example/p/1") == (
        "https://shop.example/img/saree.jpg"
    )


def test_extracts_og_image_with_reversed_attribute_order() -> None:
    """Attribute order varies between platforms."""
    html = '<meta content="https://cdn.example/a.jpg" property="og:image">'
    assert _extract_image_url(html, "https://shop.example") == "https://cdn.example/a.jpg"


def test_falls_back_through_the_metadata_candidates() -> None:
    """Twitter Card and image_src are checked when og:image is absent."""
    twitter = '<meta name="twitter:image" content="https://cdn.example/t.jpg">'
    assert _extract_image_url(twitter, "https://shop.example") == "https://cdn.example/t.jpg"

    link = '<link rel="image_src" href="https://cdn.example/l.jpg">'
    assert _extract_image_url(link, "https://shop.example") == "https://cdn.example/l.jpg"


def test_og_image_is_preferred_over_twitter() -> None:
    """When both exist, the Open Graph one is the primary product image."""
    html = (
        '<meta name="twitter:image" content="https://cdn.example/t.jpg">'
        '<meta property="og:image" content="https://cdn.example/og.jpg">'
    )
    assert _extract_image_url(html, "https://shop.example") == "https://cdn.example/og.jpg"


def test_page_without_metadata_tells_the_user_what_to_do() -> None:
    """The fallback advice must be actionable, not just 'failed'."""
    try:
        _extract_image_url("<html><body>no meta here</body></html>", "https://shop.example")
    except InvalidInputError as exc:
        assert "copy the image address" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("missing metadata accepted")


def test_relative_urls_resolve_against_the_page() -> None:
    """Shops commonly emit protocol-relative or root-relative image paths."""
    html = '<meta property="og:image" content="//cdn.example/x.jpg">'
    assert _extract_image_url(html, "https://shop.example/p/1").endswith("//cdn.example/x.jpg")


# --------------------------------------------------------------------------- #
# Live fetches — skipped offline
# --------------------------------------------------------------------------- #


def test_direct_image_url_fetches() -> None:
    """The simple case: a link straight to a JPEG."""
    if not _online():
        print("    (skipped: offline)")
        return

    fetched = fetch_image(LIVE_IMAGE_URL)
    assert isinstance(fetched, FetchedImage)
    assert not fetched.from_product_page
    assert fetched.image.mode == "RGB"
    assert min(fetched.image.size) > 100
    assert fetched.size_bytes > 1000
    assert "Loaded" in fetched.describe()


def test_product_page_yields_its_main_image() -> None:
    """The flow the user actually asked for: paste a page link."""
    if not _online("en.wikipedia.org"):
        print("    (skipped: offline)")
        return

    fetched = fetch_image(LIVE_PAGE_URL)
    assert fetched.from_product_page
    assert fetched.image.mode == "RGB"
    assert min(fetched.image.size) > 100
    assert "product page" in fetched.describe()


def test_product_page_can_be_disallowed() -> None:
    """``allow_product_page=False`` must require a direct image."""
    if not _online("en.wikipedia.org"):
        print("    (skipped: offline)")
        return

    try:
        fetch_image(LIVE_PAGE_URL, allow_product_page=False)
    except InvalidInputError as exc:
        assert "not an image" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("HTML accepted with allow_product_page=False")


def test_non_image_content_is_rejected() -> None:
    """A text file must be refused with advice to upload instead."""
    if not _online("www.rfc-editor.org"):
        print("    (skipped: offline)")
        return

    try:
        fetch_image("https://www.rfc-editor.org/rfc/rfc2616.txt")
    except InvalidInputError as exc:
        assert "not an image" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("text/plain accepted as an image")


def test_fetched_image_feeds_the_preprocessing_pipeline() -> None:
    """A fetched garment must flow straight into the real pipeline."""
    if not _online():
        print("    (skipped: offline)")
        return

    from ai_trial_room.config import Category
    from ai_trial_room.preprocessing.garment import prepare_garment

    fetched = fetch_image(LIVE_IMAGE_URL)
    assets = prepare_garment(fetched.image, Category.SAREE)

    assert assets.image.size == tuple(
        __import__("ai_trial_room.config", fromlist=["CONFIG"]).CONFIG.runtime.size
    )
    assert min(assets.cutout.size) > 32


# --------------------------------------------------------------------------- #
# UI handler
# --------------------------------------------------------------------------- #


def test_ui_handler_leaves_the_image_alone_on_failure() -> None:
    """A bad paste must not wipe a garment the user already uploaded."""
    if importlib.util.find_spec("gradio") is None:
        print("    (skipped: gradio not installed)")
        return

    import app

    update, status = app.on_garment_url("http://127.0.0.1/x.jpg")
    assert "value" not in update, "must not overwrite the existing upload"
    assert status.startswith("⚠️")
    assert "private address" in status


def test_ui_handler_ignores_an_empty_box() -> None:
    """Clicking Load with nothing pasted should be a silent no-op."""
    if importlib.util.find_spec("gradio") is None:
        print("    (skipped: gradio not installed)")
        return

    import app

    update, status = app.on_garment_url("")
    assert "value" not in update
    assert status == ""


def test_ui_handler_sets_the_image_on_success() -> None:
    """A good link must populate the garment component."""
    if importlib.util.find_spec("gradio") is None or not _online():
        print("    (skipped: gradio missing or offline)")
        return

    import app

    update, status = app.on_garment_url(LIVE_IMAGE_URL)
    assert isinstance(update.get("value"), Image.Image)
    assert status.startswith("✅")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _main() -> int:
    """Run every ``test_*`` in this module."""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures: list[tuple[str, BaseException]] = []

    for name, func in tests:
        try:
            func()
        except BaseException as exc:  # noqa: BLE001 - this *is* the runner
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from ai_trial_room.utils.logging_setup import setup_logging

    setup_logging("ERROR")
    raise SystemExit(_main())
