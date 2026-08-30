import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="FlowerVision AI",
    page_icon="🌸",
    layout="wide"
)


# ==========================
# Theme
# ==========================

st.markdown(
"""
<style>


.stApp {

    background:
    linear-gradient(
        135deg,
        #f3fff5,
        #fff7ef
    );

}


/* All text */

html, body, p, span, label, div {

    color:#263238;

}


/* Streamlit text */

.stMarkdown {

    color:#263238;

}


/* Header */

h1 {

    color:#2e7d32 !important;

    text-align:center;

    font-size:50px;

    font-weight:800;

}


.subtitle {

    text-align:center;

    color:#546e7a !important;

    font-size:21px;

}



/* Cards */


.card {


    background-color:#ffffff;


    padding:30px;


    border-radius:25px;


    box-shadow:

    0px 10px 35px rgba(46,125,50,0.18);


    border:1px solid #dcedc8;


}



/* Card title */


.card-title {


    color:#2e7d32 !important;


    font-size:26px;


    font-weight:800;


}



/* Prediction */


.result {


    text-align:center;


    color:#c2185b !important;


    font-size:38px;


    font-weight:900;


}



/* Info section */


.info {


    color:#37474f !important;


    font-size:18px;


    line-height:2;


}



.info b {


    color:#1b5e20 !important;

}



/* Top predictions */


.stProgress > div > div > div > div {

    background:#43a047;

}


/* File uploader */


[data-testid="stFileUploader"] {


    background:white;

    padding:15px;

    border-radius:15px;

}



/* Metrics */

[data-testid="stMetricValue"] {

    color:#2e7d32 !important;

}



</style>
""",
unsafe_allow_html=True
)



# ==========================
# Classes
# IMPORTANT:
# Must be exactly train_ds.class_names
# ==========================

CLASS_NAMES = [
    "bluebell",
    "buttercup",
    "colts_foot",
    "cowslip",
    "crocus",
    "daffodil",
    "daisy",
    "dandelion",
    "fritillary",
    "iris",
    "lily_valley",
    "pansy",
    "snowdrop",
    "sunflower",
    "tigerlily",
    "tulip"
]



# ==========================
# Load Model
# ==========================


@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "best_flower_model.keras"
    )

    return model


model = load_model()



# ==========================
# Prediction
# ==========================


def predict(image):

    image = image.resize(
        (224,224)
    )


    image = np.array(image)


    image = np.expand_dims(
        image,
        axis=0
    )


    image = preprocess_input(
        image
    )


    prediction = model.predict(
        image,
        verbose=0
    )


    return prediction[0]



# ==========================
# Header
# ==========================


st.markdown(
"""
<h1>
🌸 FlowerVision AI
</h1>


<p class="subtitle">

🌿 AI powered flower recognition greenhouse

<br>

Deep Learning | VGG16 | Fine-Tuning

</p>

""",
unsafe_allow_html=True
)



st.divider()



# ==========================
# Main
# ==========================


left,right = st.columns(
    [1,1]
)



with left:


    st.markdown(
    """
    <div class="card">

    <div class="card-title">
    🌱 Upload Flower Image
    </div>

    </div>
    """,
    unsafe_allow_html=True
    )


    uploaded = st.file_uploader(
        "Choose image",
        type=["jpg","jpeg","png"]
    )


    if uploaded:


        image = Image.open(
            uploaded
        ).convert("RGB")


        st.image(
            image,
            caption="Uploaded Flower",
            width=350
        )



with right:


    st.markdown(
    """
    <div class="card">

    <div class="card-title">
    🤖 AI Prediction
    </div>

    """,
    unsafe_allow_html=True
    )


    if uploaded:


        pred = predict(image)


        index = np.argmax(pred)


        confidence = pred[index]


        flower = CLASS_NAMES[index]



        st.markdown(
        f"""

        <div class="result">

        🌸 {flower.replace("_"," ").title()}

        </div>

        """,
        unsafe_allow_html=True
        )


        st.write(
            f"Confidence: {confidence*100:.2f}%"
        )


        st.progress(
            float(confidence)
        )


        st.divider()


        st.subheader(
            "🌺 Top Predictions"
        )


        top3 = np.argsort(pred)[::-1][:3]


        for rank, i in enumerate(top3, start=1):

            st.markdown(
    f"""
    🌸 **#{rank} {CLASS_NAMES[i].replace("_"," ").title()}**

    Confidence:
    {pred[i]*100:.2f}%

    ---
    """
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )



# ==========================
# Model Info
# ==========================


st.divider()


st.markdown(
"""
<div class="card">


<div class="card-title">

🌿 About FlowerVision AI

</div>


<div class="info">


<b>Architecture:</b>

VGG16


<br>


<b>Training:</b>

ImageNet Transfer Learning + Fine-Tuning


<br>


<b>Input:</b>

224 × 224 RGB Image


<br>


<b>Classes:</b>

17 Flower Species


<br>


<b>Framework:</b>

TensorFlow / Keras


</div>


</div>

""",
unsafe_allow_html=True
)