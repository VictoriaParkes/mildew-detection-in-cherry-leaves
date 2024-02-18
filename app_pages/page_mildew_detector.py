import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)


def page_mildew_detector_body():
    st.success(
        f"ML system that is capable of predicting whether a cherry leaf is "
        f"healthy or contains powdery mildew.\n\n"
        f"**Hypothesis**: Using the ML system developed for this project, "
        f"Farmy and Food's employee's will not need to have the botanical "
        f"expertise needed to diagnose powdery mildew on cherry leaf samples "
        f"to correctly identify infected trees.\n\n"
        f"**How to validate**: Neural Networks can be used to map the relationships between the features and the labels of a dataset containing images of known examples of healthy and mildew affected cherry leaves, and develop a binary classifier that will predict cherry leaf image classification over real-time data."
    )

    st.info(
        f"**Collect Samples of Suspected Powdery Mildew Containing Leaves** Look for early leaf infections on root suckers, the interior of the canopy or the crotch of the tree where humidity is high. The disease is more likely to initiate on the undersides (abaxial) of leaves (Fig. 2) but will occur on both sides at later stages. As the season progresses and infection is spread by wind, leaves may become distorted, curling upward. Severe infections may cause leaves to pucker and twist ([Claudia Probst and Gary Grove (WSU Plant Pathology), Cherry Powdery Mildew](https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/)).\n\n"
        f"Alternatively, download a set of cherry leaf images for live prediction from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader("Upload a clear picture of a cherry leaf. You may select more than one.",type=["png", "jpg"],accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Cherry Leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = "v2"
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append({"Name":image.name, 'Result': pred_class },ignore_index=True)
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)