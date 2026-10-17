
import os
import tempfile
import streamlit as st

from inference import run_inference

st.set_page_config(
    page_title="PREVIS --- Multimodal Banana Freshness & Spoilage AI Engine",
    page_icon="",
    layout="wide"
)

st.title(" PREVIS")
st.subheader("Predict food spoilage before it becomes visible")

st.write(
    "**PREVIS** is an experimental multimodal AI prototype for studying banana freshness "
    "and spoilage progression using infrared-fusion images and environmental/gas-sensor measurements[cite: 1]. "
    "The system combines visual patterns extracted from IR-fusion images with processed gas features and session metadata "
    "to estimate spoilage risks and time-to-decay milestone thresholds (such as 50%, 75%, 90%, and 95% spoiled)[cite: 1]."
)

st.warning(
    "Research prototype: predictions are estimates from a trained model "
    "and should not be used alone to determine food safety."
)

st.divider()

st.header("1. Upload infrared image")

uploaded_file = st.file_uploader(
    "Choose an IR image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded infrared image",
        width=350
    )

st.header("2. Enter sensor readings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Methane readings")

    methane_left = st.number_input(
        "Left sensor (ppm)",
        min_value=0.0,
        value=15.0,
        step=0.1
    )

    methane_right = st.number_input(
        "Right sensor (ppm)",
        min_value=0.0,
        value=16.0,
        step=0.1
    )

with col2:
    st.subheader("Temperature readings")

    temp1 = st.number_input(
        "Temperature 1 (°C)",
        value=29.5,
        step=0.1
    )

    temp2 = st.number_input(
        "Temperature 2 (°C)",
        value=31.5,
        step=0.1
    )

st.caption(
    "The two gas standard deviation features currently use fixed "
    "placeholder values in the inference pipeline."
)

if st.button("Predict Freshness", type="primary", use_container_width=True):

    if uploaded_file is None:
        st.error("Please upload an infrared image first.")

    else:
        temp_path = None

        try:
            suffix = os.path.splitext(uploaded_file.name)[1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            with st.spinner("Analyzing image and sensor readings..."):

                result = run_inference(
                    image_path=temp_path,
                    methane_ppm_left=methane_left,
                    methane_ppm_right=methane_right,
                    temp1=temp1,
                    temp2=temp2
                )

            if result is None:
                st.error(
                    "The model bundle was not found. "
                    "Check that model_bundle.pkl is available."
                )

            else:
                st.divider()
                st.header("3. Prediction results")

                score = result["freshness_score"]
                remaining = result["spoil_95_days"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Model freshness score",
                        f"{score:.1f}%"
                    )
                    st.progress(
                        max(0.0, min(score / 100.0, 1.0))
                    )

                with col2:
                    st.metric(
                        "Estimated days to 95% threshold",
                        f"{remaining:.2f} days"
                    )

                st.subheader("Estimated spoilage timeline")

                labels = {
                    "50% threshold": result["spoil_50_days"],
                    "75% threshold": result["spoil_75_days"],
                    "90% threshold": result["spoil_90_days"],
                    "95% threshold": result["spoil_95_days"]
                }

                for label, days in labels.items():
                    st.write(
                        f"**{label}:** {days:.2f} days"
                    )

                st.caption(
                    "These values are model estimates, not verified "
                    "food-safety or shelf-life guarantees."
                )

        except Exception as e:
            st.error(
                "Prediction failed. Check the model files, input image "
                "format, dependencies, and application logs."
            )
            st.exception(e)

        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)
