import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_data_visualization_body():
    st.write("### Leaf Visualization Study")

    st.success(
        f"A study conducted using conventional data analysis showing how to "
        f"visually differentiate a cherry leaf that is healthy from one that "
        f"contains powdery mildew."
    )

    st.info(
        f"**Identification of Powdery Mildew Infection**\n"
        f"Initial symptoms, often occurring 7 to 10 days after the onset of "
        f"the first irrigation, are light roughly-circular, powdery looking "
        f"patches on young, susceptible leaves (newly unfolded, and light "
        f"green expanding leaves). Older leaves develop an age-related "
        f"(ontogenic) resistance to powdery mildew and are naturally more "
        f"resistant to infection than younger leaves.\n\n"
        f"The disease is more likely to initiate on the undersides (abaxial) "
        f"of leaves but will occur on both sides at later stages. As "
        f"the season progresses and infection is spread by wind, leaves may "
        f"become distorted, curling upward. Severe infections may cause "
        f"leaves to pucker and twist. Newly developed leaves on new shoots "
        f"become progressively smaller, are often pale and may be distorted "
        f"([Claudia Probst and Gary Grove (WSU Plant Pathology), Cherry "
        f"Powdery Mildew](https://treefruit.wsu.edu/crop-protection/disease-"
        f"management/cherry-powdery-mildew/)).\n\n"

        f"**Summary of Symptoms**\n\n"
        f"* Yellowing or distortion of leaves\n"
        f"* Stunted shoot growth\n"
        f"* Reduced yield\n"
        f"* White powdery residue, which is a mixture of the fungal mycelium "
        f"and spores on leaves and fruit."
    )

    version = "v3"
    if st.checkbox("Difference between average and variability image"):
        avg_healthy = plt.imread(
            f"outputs/{version}/avg_var_healthy.png"
        )
        avg_powdery_mildew = plt.imread(
            f"outputs/{version}/avg_var_powdery_mildew.png"
        )

        st.warning(
            f"Studying the average and variability of images per label "
            f"highlighted that mildew infected cherry leaves exhibit more "
            f"variation across the surface of the leaf. However, the study "
            f"did not highlight any distinct patterns that could be used to "
            f"intuitively differentiate between healthy and infected leaves."
        )

        st.image(
            avg_healthy, caption="Healthy Leaf - Average and Variability"
        )
        st.image(
            avg_powdery_mildew, caption="Powdery Mildew Containing Leaf - "
            f"Average and Variability"
        )
        st.write("---")

    if st.checkbox(
        "Differences between average healthy and average powdery mildew "
        "cherry leaves"
    ):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            f"Studying the difference between average healthy and average "
            f"powdery mildew cherry leaves did not highlight patterns where "
            f"we could intuitively differentiate one from another."
        )
        
        st.image(
            diff_between_avgs, caption="Difference between average images"
        )
        st.write("---")
    
    def image_montage(
        dir_path, label_to_display, nrows, ncols, figsize=(15,10)
    ):
        sns.set_style("white")
        labels = os.listdir(dir_path)

        # subset the class you are interested to display
        if label_to_display in labels:

            # checks if your montage space is greater than subset size
            # how many images in that folder
            images_list = os.listdir(dir_path+'/'+ label_to_display)
            if nrows * ncols < len(images_list):
                img_idx = random.sample(images_list, nrows * ncols)
            else:
                print(
                    f"Decrease nrows or ncols to create your montage. \n"
                    f"There are {len(images_list)} in your subset. "
                    f"You requested a montage with {nrows * ncols} spaces"
                )
                return

            # create list of axes indices based on nrows and ncols
            list_rows= range(0,nrows)
            list_cols= range(0,ncols)
            plot_idx = list(itertools.product(list_rows,list_cols))

            # create a Figure and display images
            fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
            for x in range(0,nrows*ncols):
                img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
                img_shape = img.shape
                axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
                axes[plot_idx[x][0], plot_idx[x][1]].set_title(
                    f"Width {img_shape[1]}px x Height {img_shape[0]}px"
                )
                axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
                axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
            plt.tight_layout()

            st.pyplot(fig=fig)
            # plt.show()

        else:
            print("The label you selected doesn't exist.")
            print(f"The existing options are: {labels}")

    if st.checkbox("Image Montage"):
        st.warning(
            f"The image montage can be used to visually identify differences "
            f"between a healthy leaf and a mildew infected one, and highlight "
            f"typical signs of mildew infection."
        )
        st.write(
            "* To refresh the montage, click on the 'Create Montage' button"
        )
        my_data_dir = 'inputs/dataset/cherry-leaves'
        labels = os.listdir(my_data_dir+ '/validation')
        label_to_display = st.selectbox(
            label="Select label", options=labels, index=0
        )
        if st.button("Create Montage"):      
            image_montage(
                dir_path= my_data_dir + '/validation',
                label_to_display=label_to_display,nrows=8,
                ncols=3,
                figsize=(10,25)
            )
        st.write("---")