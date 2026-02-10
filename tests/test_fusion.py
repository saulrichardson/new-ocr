from __future__ import annotations

from newsbag.fusion import dedupe_boxes


def test_dedupe_boxes_drops_large_unsupported_text_strip() -> None:
    # Page is tall; a thin full-width strip is common "bad" output for some parsers.
    w, h = 2000, 3000
    page_area = float(w * h)
    line_cover_threshold = 0.50

    # Pseudo-lines: a few "real" lines below the strip.
    pseudo_lines = [
        {"bbox_xyxy": [100, 400 + i * 40, 1900, 420 + i * 40], "text": "x", "score": 1.0}
        for i in range(6)
    ]

    bad_strip = {
        "source_family": "mineru",
        "source_model": "mineru25",
        "source_label": "text",
        "norm_label": "text",
        "bbox_xyxy": [0, 40, 2000, 160],  # very wide + very short
        "score": 0.2,
        "reading_order": None,
        "text": None,
    }
    good_text = {
        "source_family": "paddle",
        "source_model": "pld_v3_thr03",
        "source_label": "text",
        "norm_label": "text",
        "bbox_xyxy": [80, 380, 1920, 900],
        "score": 0.9,
        "reading_order": None,
        "text": None,
    }

    fused = dedupe_boxes(
        candidates=[bad_strip, good_text],
        pseudo_lines=pseudo_lines,
        page_area=page_area,
        page_w=w,
        page_h=h,
        line_cover_threshold=line_cover_threshold,
    )

    # The strip should be suppressed; we should still keep the good region.
    assert all(b["bbox_xyxy"] != bad_strip["bbox_xyxy"] for b in fused)
    # For large, weakly-supported text regions we prefer synthetic recovery blocks.
    assert any(b.get("source_family") == "synthetic" for b in fused)


def test_dedupe_boxes_replaces_large_unsupported_text_block_with_recovery_boxes() -> None:
    w, h = 2400, 3200
    page_area = float(w * h)
    line_cover_threshold = 0.50

    # Many pseudo-lines inside the big block.
    pseudo_lines = [
        {"bbox_xyxy": [150, 900 + i * 35, 2250, 920 + i * 35], "text": "x", "score": 1.0}
        for i in range(20)
    ]

    giant = {
        "source_family": "dell",
        "source_model": "dell_c0005_i010",
        "source_label": "article",
        "norm_label": "text",
        "bbox_xyxy": [50, 800, 2350, 3100],  # very large block
        "score": 0.4,
        "reading_order": None,
        "text": None,
    }

    fused = dedupe_boxes(
        candidates=[giant],
        pseudo_lines=pseudo_lines,
        page_area=page_area,
        page_w=w,
        page_h=h,
        line_cover_threshold=line_cover_threshold,
    )

    # We should not keep the raw giant block; we should get smaller synthetic recovery boxes.
    assert all(b["bbox_xyxy"] != giant["bbox_xyxy"] for b in fused)
    assert any(b.get("source_family") == "synthetic" for b in fused)
