"""Batch Detection: many images, processed independently, with a combined summary and a ZIP download."""
import hashlib

import pandas as pd
import streamlit as st

from src.app.common import (APP_OUTPUT_DIR, counts_chart, current_settings, demo_images, get_engine, timestamp)
from src.inference.batch_processor import MAX_BATCH_IMAGES, BatchResult, process_batch, save_batch
from src.inference.image_processor import METHOD_LABELS, SUPPORTED_FORMATS, SUPPORTED_FORMATS_TEXT, YOLO
from src.utils.helpers import PROJECT_ROOT


def _signature(files: list[tuple[str, bytes]], settings: dict) -> str:
    h = hashlib.sha256()
    for name, data in files:
        h.update(name.encode())
        h.update(hashlib.sha256(data).digest())
    h.update(repr(sorted(settings.items())).encode())
    return h.hexdigest()


def render() -> None:
    settings = current_settings()
    method = settings["method"]
    st.title("Batch Detection")
    st.caption(f"Model: **{METHOD_LABELS[method]}**. Each image is processed independently; unreadable files are "
               f"reported and skipped. Up to {MAX_BATCH_IMAGES} images per batch.")

    use_demo = st.toggle("Use the 8 demonstration images instead of uploading", key="batch_demo")
    if use_demo:
        files = [(p.name, p.read_bytes()) for p in demo_images()]
        st.caption("Demonstration data: photos from the dataset's test split (CC0 license).")
    else:
        uploads = st.file_uploader(f"Upload images ({SUPPORTED_FORMATS_TEXT})", type=list(SUPPORTED_FORMATS),
                                   accept_multiple_files=True, key="batch_upload")
        files = [(f.name, f.getvalue()) for f in uploads or []]
    if not files:
        st.info("Select several images to process them in one go.", icon=":material/upload:")
        return

    sig = _signature(files, settings)
    if st.button(f"Detect fruits in {len(files)} image(s)", type="primary", icon=":material/play_arrow:"):
        bar = st.progress(0.0, text="Starting ...")
        try:
            batch = process_batch(files, get_engine(), method,
                                  settings["conf"] if method == YOLO else None,
                                  settings["iou"] if method == YOLO else None,
                                  settings["imgsz"] if method == YOLO else None, settings["show_conf"],
                                  progress=lambda k, n, name: bar.progress(k / n, text=f"{k}/{n}: {name}"))
        except ValueError as exc:
            bar.empty()
            st.error(str(exc))
            return
        bar.empty()
        st.session_state["batch_result"] = (sig, batch)

    stored = st.session_state.get("batch_result")
    if not stored:
        return
    stored_sig, batch = stored
    if stored_sig != sig:
        st.info("The images or settings changed since the last run. Press the button to process them again.",
                icon=":material/refresh:")
        return
    _show_batch(batch)


def _show_batch(batch: BatchResult) -> None:
    t = batch.timing_summary()
    m = st.columns(5)
    m[0].metric("Images processed", len(batch.succeeded))
    m[1].metric("Failed", len(batch.failed))
    m[2].metric("Total fruits", batch.total_fruits)
    m[3].metric("Mean time / image", f"{t['mean_ms']:.0f} ms" if t["mean_ms"] is not None else "–")
    m[4].metric("Batch wall time", f"{t['wall_time_ms'] / 1000:.1f} s")

    for item in batch.failed:
        st.error(item.error, icon=":material/broken_image:")
    notices = {i.notice for i in batch.succeeded if i.notice}
    for n in notices:
        st.warning(n, icon=":material/warning:")

    c1, _ = st.columns([1, 1])
    with c1:
        st.markdown("**Combined fruit count**")
        st.altair_chart(counts_chart(batch.combined_counts()), width="stretch")
    st.markdown("**Per-image summary**")
    st.dataframe(pd.DataFrame(batch.rows()).drop(columns=["Method"]), hide_index=True, width="stretch",
                 height=min(38 + 35 * len(batch.items), 400),
                 column_config={"Time (ms)": st.column_config.NumberColumn(format="%.0f")})

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download all results (ZIP)", batch.to_zip(), f"bototoy_batch_{timestamp()}.zip",
                       "application/zip", icon=":material/folder_zip:", width="stretch",
                       help="Annotated images (longest side up to 2048 px) + results.csv + summary.json "
                            "(all detections, boxes in original-image pixels).")
    d2.download_button("Download table (CSV)", batch.to_csv(), "bototoy_batch_results.csv", "text/csv",
                       icon=":material/table:", width="stretch")
    if d3.button("Save to outputs/app/", icon=":material/save:", width="stretch", key="batch_save"):
        out = save_batch(batch, APP_OUTPUT_DIR / f"batch_{timestamp()}")
        st.success(f"Saved to {out.relative_to(PROJECT_ROOT).as_posix()}/", icon=":material/check_circle:")

    st.markdown("**Annotated images**")
    ok = batch.succeeded
    for start in range(0, len(ok), 3):
        cols = st.columns(3)
        for col, item in zip(cols, ok[start:start + 3]):
            with col.container(border=True):
                st.image(item.annotated_jpeg, width="stretch")
                found = ", ".join(f"{n} {c}" for c, n in item.counts.items() if n) or "no fruit detected"
                st.markdown(f"**{item.name}**  \n{item.total} fruit(s): {found}  \n"
                            f"{item.timing_ms['end_to_end']:.0f} ms · {METHOD_LABELS[item.method]}")
