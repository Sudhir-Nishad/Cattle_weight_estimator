"""
Cattle Weight Estimation — Streamlit Web App
------------------------------------------------------------------------------
Workflow:
  1. User enters a Tag ID and uploads a SIDE-VIEW photo of the animal
     (calibration sticker must be visible).
  2. Segmentation model (best.pt) locates the cow + sticker for calibration.
  3. Keypoint model (best_model_side.pth) locates side-view keypoints.
  4. Body Length and Heart Girth are computed, then:
         Weight (kg) = Heart_Girth^2 * Body_Length / 10840
  5. Weight is shown on screen, with:
        - "View Output Image" button -> opens a dialog with the annotated
          image (Body Length + Heart Girth only) and a download button.
        - "New Estimation" button -> fully clears the Tag ID + uploaded
          image + result and scrolls back to the top of the page.
  6. Sidebar "View All Logs" button opens a dialog listing every past
     measurement as an expandable card per Tag ID + timestamp, with an
     "Download Excel" button for the full log.
==============================================================================
"""

import os
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from utils import (
    load_models,
    run_side_inference,
    compute_weight_traits,
    draw_weight_annotation,
    STICKER_CM_DEFAULT,
)

# ==============================================================================
# Page config + branding
# ==============================================================================
LOGO_PATH = "assets/logo.png"
LOGS_PATH = "logs/cattle_weight_logs.csv"
LOG_COLUMNS = [
    "Tag_ID", "Date", "Time",
    "Linear_Body_Depth_cm", "Linear_Chest_Height_cm",
    "Body_Length_cm", "Heart_Girth_cm", "Weight_kg",
]

PRIMARY = "#2A2866"

st.set_page_config(
    page_title="Cattle Weight Estimator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    /* Use Streamlit CSS variables so colors adapt automatically to dark & light modes */
    h1, h2, h3 {{
        color: var(--text-color);
    }}

    /* --------------------------------------------------------------------
       Sidebar background — pinned on every wrapper Streamlit might use for
       the sidebar panel, with a hard fallback color (not just the CSS
       variable) so it can never resolve to the browser's transparent
       default. On mobile the sidebar renders as a sliding overlay drawer;
       if only the outer <section> got a background (and not its inner
       content wrapper), the drawer's empty space shows the page behind it
       through, which is the "transparent sidebar" bug.
       -------------------------------------------------------------------- */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
        background-color: var(--secondary-background-color, #F4F3FA) !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] {{
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }}
    /* Make sure the drawer covers the full scrollable height on mobile,
       not just the initial viewport, so nothing peeks out below the fold. */
    @media (max-width: 767px) {{
        section[data-testid="stSidebar"] {{
            min-height: 100vh !important;
            min-height: 100dvh !important;
        }}
    }}

    /* Sidebar logo — fixed, modest size on every device, left-aligned so
       it lines up with the rest of the sidebar content (button, caption). */
    section[data-testid="stSidebar"] img {{
        max-width: 150px !important;
        width: 100% !important;
        height: auto !important;
        margin: 0 !important;
        display: block;
    }}
    /* The element wrapping st.image also gets centered by Streamlit by
       default — force it back to flush-left so it matches the button. */
    section[data-testid="stSidebar"] [data-testid="stImage"] {{
        display: flex !important;
        justify-content: flex-start !important;
    }}
    /* Sidebar buttons — capped to same width as the logo, regardless of
       sidebar width, and left-aligned (not centered) so it lines up with
       the logo above it instead of floating in the middle. */
    section[data-testid="stSidebar"] div.stButton {{
        display: flex !important;
        justify-content: flex-start !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button {{
        max-width: 150px !important;
        width: 100% !important;
        margin: 0 !important;
        font-size: 13px;
        padding: 0.4em 0.7em;
        border-radius: 6px;
    }}

    /* Lock sidebar width ONLY on Desktop screens (>= 768px) so mobile drawer isn't cut off */
    @media (min-width: 768px) {{
        section[data-testid="stSidebar"] {{
            min-width: 260px !important;
            max-width: 260px !important;
            width: 260px !important;
        }}
    }}

    [data-testid="stSidebarResizeHandle"] {{
        display: none !important;
        pointer-events: none !important;
        width: 0 !important;
    }}
    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {{
        font-size: 12px;
    }}
    div.stButton > button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.4em;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        background-color: #201d52;
        color: white;
    }}
    div.stDownloadButton > button {{
        background-color: transparent;
        color: var(--text-color);
        border: 1.5px solid {PRIMARY};
        border-radius: 8px;
        font-weight: 600;
    }}
    div.stDownloadButton > button:hover {{
        background-color: {PRIMARY};
        color: white;
    }}
    .weight-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #46418f 100%);
        color: #FFFFFF !important;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    .weight-card .value {{
        font-size: 52px;
        font-weight: 800;
        line-height: 1.1;
        color: #FFFFFF !important;
    }}
    .weight-card .label {{
        font-size: 15px;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.85;
        color: #FFFFFF !important;
    }}
    .metric-box {{
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }}
    .metric-box .val {{
        font-size: 22px;
        font-weight: 700;
        color: var(--text-color);
    }}
    .metric-box .lab {{
        font-size: 12px;
        color: var(--text-color);
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .log-banner {{
        background: linear-gradient(90deg, {PRIMARY} 0%, #55519e 55%, #cfcfe6 100%);
        padding: 14px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 14px;
    }}
    footer {{visibility: hidden;}}

    /* Extra shrink on very narrow (phone) screens */
    @media (max-width: 400px) {{
        section[data-testid="stSidebar"] img {{
            max-width: 150px !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button {{
            max-width: 150px !important;
            margin: 0 !important;
            font-size: 12px;
            padding: 0.35em 0.6em;
        }}
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# Session state
# ==============================================================================
DEFAULTS = {
    "logs_df": None,
    "form_version": 0,
    "estimation_done": False,
    "last_annotated_img": None,
    "last_tag_id": None,
    "last_traits": None,
    "last_warning": None,
    "scroll_to_top": False,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.logs_df is None:
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(LOGS_PATH):
        st.session_state.logs_df = pd.read_csv(LOGS_PATH)
    else:
        st.session_state.logs_df = pd.DataFrame(columns=LOG_COLUMNS)

# ── Scroll-to-top ──
if st.session_state.scroll_to_top:
    components.html(
        """
        <script>
        (function() {
            function scrollToTop() {
                try {
                    var doc = window.parent.document;
                    var anchor = doc.getElementById('page-top');
                    if (anchor && anchor.scrollIntoView) {
                        anchor.scrollIntoView({behavior: 'auto', block: 'start', inline: 'nearest'});
                    }
                    var targets = [
                        doc.querySelector('[data-testid="stAppViewContainer"]'),
                        doc.querySelector('[data-testid="stMainBlockContainer"]'),
                        doc.querySelector('section.main'),
                        doc.querySelector('.main'),
                        doc.scrollingElement,
                        doc.documentElement,
                        doc.body,
                    ];
                    targets.forEach(function(el) {
                        if (!el) return;
                        if (typeof el.scrollTo === 'function') {
                            el.scrollTo({top: 0, left: 0, behavior: 'auto'});
                        } else {
                            el.scrollTop = 0;
                        }
                    });
                    window.parent.scrollTo({top: 0, left: 0, behavior: 'auto'});
                } catch (e) {}
            }
            var attempts = 0;
            var interval = setInterval(function() {
                scrollToTop();
                attempts++;
                if (attempts > 30) clearInterval(interval);
            }, 100);
        })();
        </script>
        """,
        height=0,
    )
    st.session_state.scroll_to_top = False


# ==============================================================================
# Model loading
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_models():
    status = st.empty()

    def progress(msg):
        status.info(msg)

    yolo_model, resnet_model, device = load_models(progress_callback=progress)
    status.empty()
    return yolo_model, resnet_model, device


# ==============================================================================
# Helpers
# ==============================================================================
def reset_form():
    """Fully clears Tag ID + uploaded image + current result, then scrolls to top."""
    st.session_state.form_version += 1
    st.session_state.estimation_done = False
    st.session_state.last_annotated_img = None
    st.session_state.last_tag_id = None
    st.session_state.last_traits = None
    st.session_state.last_warning = None
    st.session_state.scroll_to_top = True
    st.rerun()


def render_measurement_table_html(measurements: dict) -> str:
    rows_html = "".join(
        f"<tr>"
        f"<td style='padding:9px 14px;border-bottom:1px solid rgba(128,128,128,0.2);color:var(--text-color);font-weight:600;'>{k}</td>"
        f"<td style='padding:9px 14px;border-bottom:1px solid rgba(128,128,128,0.2);color:var(--text-color);'>{v}</td>"
        f"</tr>"
        for k, v in measurements.items()
    )
    return f"""
    <table style="width:100%; border-collapse:collapse; margin-bottom:6px;">
        <thead>
            <tr style="background-color:{PRIMARY}; color:white;">
                <th style="padding:10px 14px; text-align:left; font-size:12px; letter-spacing:0.5px;">MEASUREMENT</th>
                <th style="padding:10px 14px; text-align:left; font-size:12px; letter-spacing:0.5px;">VALUE</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def logs_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cattle Weight Logs")
    return buf.getvalue()


# ==============================================================================
# Dialogs
# ==============================================================================
@st.dialog("Output Image", width="large")
def show_output_dialog():
    img = st.session_state.last_annotated_img
    tag = st.session_state.last_tag_id
    if img is None:
        st.info("No image available yet.")
        return

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption=f"Tag ID: {tag}",
        use_container_width=True,
    )

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    safe_tag = (tag or "cattle").replace(" ", "_")
    st.download_button(
        "⬇ Download annotated image",
        data=buf.getvalue(),
        file_name=f"{safe_tag}_annotated.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )


@st.dialog("Measurement Logs", width="large")
def show_logs_dialog():
    st.markdown(
        '<div class="log-banner">📋 Measurement Logs — All Cattle</div>',
        unsafe_allow_html=True,
    )

    df = st.session_state.logs_df

    top_l, top_r = st.columns([2, 1])
    with top_l:
        st.markdown(f"**{len(df)} record(s) in log**")
    with top_r:
        if not df.empty:
            st.download_button(
                "⬇ Download Excel",
                data=logs_to_excel_bytes(df),
                file_name="cattle_weight_logs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    if df.empty:
        st.info("No logs yet. Estimate a cattle's weight to start logging.")
        return

    search = st.text_input("🔍 Search by Tag ID", key="log_search")
    view_df = df
    if search:
        view_df = df[df["Tag_ID"].astype(str).str.contains(search, case=False, na=False)]

    if view_df.empty:
        st.warning("No records match that Tag ID.")
        return

    view_df = view_df.sort_values(by=["Date", "Time"], ascending=False)

    for _, row in view_df.iterrows():
        header = f"🐄 {row['Tag_ID']}  |  {row['Date']} {row['Time']}"
        with st.expander(header):
            measurements = {
                "Linear Body Depth": f"{row['Linear_Body_Depth_cm']:.2f} cm" if pd.notna(row["Linear_Body_Depth_cm"]) else "N/A",
                "Linear Chest Height": f"{row['Linear_Chest_Height_cm']:.2f} cm" if pd.notna(row["Linear_Chest_Height_cm"]) else "N/A",
                "Body Length": f"{row['Body_Length_cm']:.2f} cm" if pd.notna(row["Body_Length_cm"]) else "N/A",
                "Heart Girth": f"{row['Heart_Girth_cm']:.2f} cm" if pd.notna(row["Heart_Girth_cm"]) else "N/A",
                "Weight": f"{row['Weight_kg']:.2f} kg" if pd.notna(row["Weight_kg"]) else "N/A",
            }
            st.markdown(render_measurement_table_html(measurements), unsafe_allow_html=True)


# ==============================================================================
# Sidebar: Logo + Logs button only
# ==============================================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=130)
    else:
        st.markdown(f"<h3>🐄 CattleWeigh</h3>", unsafe_allow_html=True)
        st.caption("Drop your logo at `assets/logo.png` to replace this placeholder.")

    st.markdown("---")

    if st.button("📋 View All Logs", use_container_width=True):
        show_logs_dialog()

    st.caption(f"{len(st.session_state.logs_df)} record(s) logged so far.")

# ==============================================================================
# Main content
# ==============================================================================
st.markdown('<div id="page-top"></div>', unsafe_allow_html=True)
st.markdown("<h1>🐄 Cattle Weight Estimator</h1>", unsafe_allow_html=True)
st.caption("Upload a side-view image to estimate live body weight from Body Length and Heart Girth.")

col_form, col_result = st.columns([1, 1.2], gap="large")

with col_form:
    st.subheader("1. Animal Details")
    tag_id = st.text_input(
        "Tag ID",
        key=f"tag_id_{st.session_state.form_version}",
        placeholder="e.g. COW-1024",
    )

    st.subheader("2. Side-View Image")
    
    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Take Photo"])

    uploaded_file = None
    camera_file = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a clear side-view photo (calibration sticker visible)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            key=f"uploader_{st.session_state.form_version}",
        )

    with tab_camera:
        camera_file = st.camera_input(
            "Capture side-view photo",
            key=f"camera_{st.session_state.form_version}",
        )
        if camera_file is not None:
            cam_bytes = np.frombuffer(camera_file.getvalue(), np.uint8)
            cam_img = cv2.imdecode(cam_bytes, cv2.IMREAD_COLOR)
            if cam_img is not None:
                h_c, w_c = cam_img.shape[:2]
                if h_c > w_c:
                    cam_img = cv2.rotate(cam_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                st.caption("📷 Rotated Landscape Image:")
                st.image(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    selected_image_file = uploaded_file or camera_file

    with st.expander("⚙ Advanced settings"):
        sticker_cm = st.number_input(
            "Calibration sticker size (cm)", min_value=1.0, max_value=50.0,
            value=STICKER_CM_DEFAULT, step=0.5,
        )
        conf_thresh = st.slider("Detection confidence threshold", 0.1, 0.9, 0.3, 0.05)

    estimate_clicked = st.button("🔍 Estimate Weight", use_container_width=True)

# ==============================================================================
# Run estimation
# ==============================================================================
if estimate_clicked:
    if not tag_id.strip():
        st.warning("Please enter a Tag ID before estimating.")
    elif selected_image_file is None:
        st.warning("Please upload or capture a side-view image before estimating.")
    else:
        with st.spinner("Loading models and analyzing image..."):
            yolo_model, resnet_model, device = get_models()

            file_bytes = np.frombuffer(selected_image_file.getvalue(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is None:
                st.error("Could not read the provided image. Please try again.")
            else:
                # Automatically convert portrait images (height > width) to landscape format
                h, w = img_bgr.shape[:2]
                if h > w:
                    img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

                inference = run_side_inference(
                    yolo_model, resnet_model, img_bgr, device,
                    sticker_cm=sticker_cm, score_thresh=conf_thresh,
                )

                if "error" in inference:
                    st.error(inference["error"])
                else:
                    traits = compute_weight_traits(inference["pred_kps"], inference["cmp"])
                    annotated = draw_weight_annotation(img_bgr, inference["pred_kps"], traits, tag_id=tag_id)

                    st.session_state.last_annotated_img = annotated
                    st.session_state.last_tag_id = tag_id.strip()
                    st.session_state.last_traits = traits
                    st.session_state.last_warning = (
                        "Some required keypoints were not detected clearly: "
                        + ", ".join(traits["missing_keypoints"])
                        + ". Results may be incomplete or unreliable."
                        if traits["missing_keypoints"] else None
                    )
                    st.session_state.estimation_done = True

                    new_row = {
                        "Tag_ID": tag_id.strip(),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Linear_Body_Depth_cm": round(traits["linear_body_depth_cm"], 2) if traits["linear_body_depth_cm"] else None,
                        "Linear_Chest_Height_cm": round(traits["linear_chest_height_cm"], 2) if traits["linear_chest_height_cm"] else None,
                        "Body_Length_cm": round(traits["body_length_cm"], 2) if traits["body_length_cm"] else None,
                        "Heart_Girth_cm": round(traits["heart_girth_cm"], 2) if traits["heart_girth_cm"] else None,
                        "Weight_kg": round(traits["weight_kg"], 2) if traits["weight_kg"] else None,
                    }
                    st.session_state.logs_df = pd.concat(
                        [st.session_state.logs_df, pd.DataFrame([new_row])],
                        ignore_index=True,
                    )
                    os.makedirs("logs", exist_ok=True)
                    st.session_state.logs_df.to_csv(LOGS_PATH, index=False)

                    st.rerun()

# ==============================================================================
# Result panel
# ==============================================================================
with col_result:
    st.subheader("Result")

    if st.session_state.estimation_done and st.session_state.last_traits is not None:
        traits = st.session_state.last_traits

        if st.session_state.last_warning:
            st.warning(st.session_state.last_warning)

        if traits["weight_kg"] is not None:
            st.markdown(
                f"""
                <div class="weight-card">
                    <div class="label">Estimated Weight</div>
                    <div class="value">{traits['weight_kg']:.1f} kg</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("Weight could not be calculated — required keypoints were not detected.")

        m1, m2 = st.columns(2)
        with m1:
            bl = traits["body_length_cm"]
            st.markdown(
                f"""<div class="metric-box"><div class="val">{f'{bl:.1f} cm' if bl else 'N/A'}</div>
                <div class="lab">Body Length</div></div>""",
                unsafe_allow_html=True,
            )
        with m2:
            hg = traits["heart_girth_cm"]
            st.markdown(
                f"""<div class="metric-box"><div class="val">{f'{hg:.1f} cm' if hg else 'N/A'}</div>
                <div class="lab">Heart Girth</div></div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🖼 View Output Image", use_container_width=True):
                show_output_dialog()
        with b2:
            if st.button("🔄 New Estimation", use_container_width=True):
                reset_form()
    else:
        st.info("Enter a Tag ID, upload a side-view image, then click **Estimate Weight**.")

st.markdown("---")
st.caption("Weight formula: (Heart Girth² × Body Length) / 10840  |  Calibration via sticker of known size in the image.")