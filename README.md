# PREVIS --- Multimodal Banana Freshness & Spoilage AI Engine

## 1. Project Overview

PREVIS is an experimental multimodal AI prototype for studying banana
freshness and spoilage progression using infrared-fusion images and
environmental/gas-sensor measurements.

The system is designed to combine visual patterns extracted from
IR-fusion JPG images with already-processed gas features and relevant
session metadata. The intended output is a research-oriented estimate of
spoilage risk and, if sufficient labeled data is available, estimated
time to defined spoilage milestones.

> **Disclaimer:** All values, estimates, and predictions are
> experimental calculations derived from the available dataset and
> laboratory conditions. PREVIS is not certified for food safety,
> health, or regulatory compliance. Its outputs must not be relied upon
> for critical safety decisions. The creators and contributors assume no
> liability for damages, losses, or claims arising from use or misuse of
> this application.

 Live Frontend Demo

A crude but functional frontend prototype is available on Streamlit. It provides a simple interface for demonstrating the banana ripeness detection concept.

**Live Demo:** [PREVIS — Banana Ripeness Detector](https://banana-ripeness-detector-bxnm5lxumaqe9oggjacg62.streamlit.app)

> **Note:** This is an early-stage demonstration frontend. Its displayed predictions and functionality should not be interpreted as scientifically validated spoilage detection unless supported by verified model outputs and experimental validation.

## 2. Problem Statement

Post-harvest food spoilage contributes to food waste and economic
losses. Conventional visual grading can miss early changes that are not
yet visible in ordinary RGB images.

PREVIS explores whether combining patterns from IR-fusion images with
gas and environmental sensor data can support non-destructive research
into banana freshness and degradation.

The prototype must distinguish between: - **Image-derived features:**
numerical patterns computed from the displayed pixels or learned by a
pretrained CNN. - **Calibrated thermal measurements:** physical
temperatures that require suitable calibrated thermal data. - **Model
predictions:** estimates that require appropriate training labels and
validation.

IR-fusion JPGs are not assumed to contain calibrated per-pixel
temperatures.

## 3. Current Data and Constraints

The current image collection is described as: - Seven days of
observations. - Three sessions per day. - A top-down view and 16
different side angles. - IR-fusion JPG images; raw thermal data is
unavailable. - Gas features have already been extracted separately. -
Ambient temperature fields such as `T1` and `T2` may be available as
metadata or sensor inputs, but they do not by themselves calibrate the
IR-fusion image into a per-pixel temperature map.

Images and sensor readings must be linked using reliable specimen, date,
session, timestamp, and view metadata. Do not treat multiple views of
the same banana at the same session as independent specimens.

## 4. Recommended Technical Approach

### 4.1 Visual IR Embedding Pipeline

Use a pretrained ResNet18 as a frozen feature extractor for the first
implementation. Remove its final classification layer and extract the
512-dimensional output vector for each image.

Recommended steps: 1. Read each IR-fusion JPG and retain its original
filename and path. 2. Validate that the file can be opened and that its
dimensions are plausible. 3. Convert the image to RGB as required by the
pretrained model's input pipeline. 4. Apply the preprocessing associated
with the selected pretrained weights. 5. Run the image through frozen
ResNet18 in inference mode. 6. Save the 512 embedding values with the
source image ID and available metadata. 7. Aggregate embeddings across
the available views for each observation session, while retaining the
per-image embeddings for traceability.

**Important:** ResNet18 was pretrained on ordinary natural images. Its
embeddings on false-color IR-fusion images may or may not be
informative. Treat them as experimental features and compare against
simple baselines when possible. The embeddings are not physical
temperature values and are not inherently spoilage probabilities.

### 4.2 Banana Localization and Masking

Masking can help exclude background pixels when calculating explicit
image statistics. Test a small sample of images from different sessions
and angles first.

Possible methods: - **Thresholding and contours:** appropriate when the
background is simple and consistent. - **Pretrained segmentation
model:** may help with complex backgrounds, but its output must be
verified on false-color IR images. - **Manual masks:** a reliable
fallback if automatic segmentation is inaccurate.

Do not assume a segmentation model will recognize the banana correctly.
Check masks visually, especially across the different side angles. A
crop or mask can also be used for CNN input, but use consistent
preprocessing across images.

### 4.3 Explicit Image Statistics (Optional)

These are separate from CNN embeddings and can be computed directly from
image pixels, preferably within a verified banana mask: - `pixel_mean` -
`pixel_std` - `thermal_p05` --- explicitly documented as an
image-intensity percentile - `thermal_p25` --- image-intensity
percentile - `thermal_p50` --- image-intensity median - `thermal_p75`
--- image-intensity percentile - `thermal_p95` --- image-intensity
percentile - `thermal_gradient` --- image-intensity gradient only, if
the method is defined - `thermal_hot_fraction` --- fraction above a
documented image-intensity threshold - `thermal_cold_fraction` ---
fraction below a documented image-intensity threshold -
`thermal_entropy` - `feature_version`

These values depend on the chosen image representation, palette,
preprocessing, mask, and thresholds. Document those choices and keep
them consistent. Do not label intensity-based features as physical
temperature.

The following physical-temperature fields should remain unavailable/null
unless calibrated thermal data is obtained: - `thermal_mean_c` -
`thermal_min_c` - `thermal_max_c` - `thermal_range_c` - `thermal_std_c`

A pixel-intensity gradient is not a physical temperature gradient.
Ambient temperature measurements alone do not supply the missing
per-pixel calibration.

### 4.4 Existing Gas Features

Gas features have already been extracted. Preserve that pipeline and its
outputs. The visual-feature task should not unnecessarily recompute gas
measurements.

Before fusion, ensure gas records are matched to the correct specimen
and observation session. Retain quality flags and missing-value
indicators where available.

### 4.5 Multimodal Fusion and Prediction

The intended high-level flow is:

``` text
IR-fusion JPG images
        |
        v
Image validation / optional banana masking
        |
        v
Frozen pretrained ResNet18
        |
        v
512-dimensional embedding per image
        |
        v
Aggregate views by specimen and session
        |
        +------------------------------+
                                       |
Existing gas features -----------------+--> Feature fusion
                                       |
Session metadata / valid labels -------+
                                       |
                                       v
                         Small predictive model
                                       |
                                       v
                 Experimental risk / timeline estimates
```

A small model such as logistic regression or a random forest can be a
baseline if suitable labels exist. A neural regression model is not
automatically justified by the size of the dataset.

## 5. Spoilage Milestones and Validation

Potential milestone outputs include estimated time until: - 50%
spoiled - 75% spoiled - 90% spoiled - 95% spoiled

These are only defensible if the project defines what each spoilage
percentage means and has suitable labeled observations to train and
evaluate the predictions. If labels or observed spoilage endpoints are
unavailable, do not present these estimates as validated or precise.

The system should: - Use specimen-level train/test separation where
possible. - Avoid splitting images from the same specimen/session across
training and test sets in a way that causes leakage. - Report
uncertainty and limitations. - Return null or an explicit unavailable
state when an estimate cannot be supported. - Avoid claims of detecting
internal microbial decay unless the dataset provides evidence validating
that interpretation.

## 6. Suggested Data Output for CNN Features

Store one record per source image, with: - `ir_input_id` - original
filename and path - specimen ID - day index - session ID and timestamp -
view angle or view identifier - `cnn_feature_000` through
`cnn_feature_511` - extraction/model version - processing status or
error information

Keep the per-image embeddings for traceability. If the prediction model
operates at session level, create a separate session-level dataset by
aggregating the available image embeddings. Document the aggregation
method (for example, element-wise mean across the views available for a
session).

Link the feature records to the original image table through
`ir_input_id`. Do not use the filename alone as a database key if a
stable image ID exists.

## 7. Repository Structure

A suggested structure is:

``` text
Banana_AI/
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── assets/
│   └── dashboard_screenshot.png
└── main/
    ├── inference.py
    ├── model_architecture.py
    ├── process_data.py
    ├── extract_ir_embeddings.py
    └── train_predictor.py
```

Roles: - `app.py`: Streamlit dashboard and input handling. -
`requirements.txt`: Python dependencies. -
`main/extract_ir_embeddings.py`: image validation, preprocessing,
ResNet18 embedding extraction, and output writing. -
`main/process_data.py`: sensor/session data preparation and
normalization. - `main/model_architecture.py`: model definitions, if a
custom fusion model is actually needed. - `main/train_predictor.py`:
training and evaluation using available labels. - `main/inference.py`:
inference orchestration and output formatting.

This is a suggested structure; keep the current repository's names if
they already serve the same purposes.

## 8. Local Installation and Launch

### Prerequisites

-   Python 3.9 or newer.
-   A virtual environment is recommended.
-   Internet access may be needed the first time pretrained model
    weights are downloaded, unless the weights are already cached or
    provided locally.

### Install dependencies

From the repository root:

``` bash
python -m venv .venv
```

Activate the environment.

On Windows:

``` bash
.venv\\Scripts\\activate
```

On macOS/Linux:

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

The image-extraction script will need compatible versions of PyTorch,
torchvision, Pillow, and pandas. Add these to `requirements.txt` if they
are not already included.

### Extract image embeddings

After configuring the image directory and metadata mapping in the
extraction script, run:

``` bash
python main/extract_ir_embeddings.py
```

The script should save a CSV or another documented tabular output
containing one row per image and 512 embedding columns, plus
image/session identifiers.

### Launch the dashboard

``` bash
streamlit run app.py
```

Then open:

``` text
http://localhost:8501
```


## 9. Limitations, Honest Claims & Future Roadmap

### Current Limitations & Honest Claims
PREVIS is an experimental research prototype with several key technical constraints:
* **Dataset Constraints & Generalizability:** The prototype is trained on a limited dataset with a small number of specimens. Consequently, prediction accuracy may vary when evaluating different bananas under unconditioned or diverse environments, and results may not generalize beyond the specific dataset, imaging setup, storage conditions, or laboratory environment.
* **Temperature & Calibration:** IR-fusion JPGs do not provide verified per-pixel temperatures without appropriate calibration, and ambient temperature sensors do not calibrate each pixel in an IR image.
* **Feature Interpretations:** CNN embeddings are learned visual representations, not direct temperature, spoilage, or microbial measurements.
* **Localization & Classification:** A pretrained ResNet18 does not automatically localize the banana or classify it as overripe, bruised, or spoiled. Localization requires a separate method, while quality classification requires relevant labeled data and a trained classifier.
* **Milestone Estimates:** Accurate time-to-spoilage estimates require suitable labels, clear milestone definitions, and validation on distinct specimens.
* **Robustness:** Model generalizability and robustness will be significantly enhanced in subsequent iterations as larger and more diverse training datasets are acquired.

### Future Scope & Expansion
* **Multi-Fruit Scalability:** While the current iteration is focused exclusively on bananas, the underlying multimodal pipeline is designed to be modular. 
* **Roadmap:** Future updates will adapt the feature architecture and parameters to support multiple different types of fruit, aiming to broaden the application scope, increase accuracy, and maximize overall model efficiency.

## 10. Summary

For the current implementation, prioritize a **frozen pretrained
ResNet18 image-embedding extractor** and leave the existing gas-feature
pipeline intact. Preserve image/session metadata, optionally calculate
explicitly defined image-intensity statistics, and combine features only
after reliable session-level alignment. Keep physical-temperature fields
unavailable unless calibrated thermal data becomes available, and
present spoilage timelines as experimental unless validated against
suitable labels.
