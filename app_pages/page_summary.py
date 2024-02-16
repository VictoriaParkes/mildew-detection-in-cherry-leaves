import streamlit as st


def page_summary_body():
    st.write("### Project Summary")

    st.info(
        f"**General Information**\n"
        f"Powdery mildew of sweet and sour cherry is caused by Podosphaera "
        f"clandestina, an obligate biotrophic fungus. Mid- and late-season "
        f"sweet cherry (Prunus avium) cultivars are commonly affected, "
        f"rendering them unmarketable due to the covering of white fungal "
        f"growth on the cherry surface ([Claudia Probst and Gary Grove (WSU "
        f"Plant Pathology), Cherry Powdery Mildew](https://treefruit.wsu.edu/"
        f"crop-protection/disease-management/cherry-powdery-mildew/)).\n\n"
        f"**Project Dataset**\n"
        f"The dataset contains +4 thousand images taken from the client's crop"
        f" fields. The images show healthy cherry leaves and cherry leaves "
        f"that have powdery mildew, a fungal disease that affects many plant "
        f"species."
    )

    st.write(
        f"* **For additional information**, please visit and read the "
        f"[Project README file](https://github.com/VictoriaParkes/mildew-"
        f"detection-in-cherry-leaves/blob/main/README.md).")
    
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in conducting a study to visually "
        f"differentiate a healthy cherry leaf from one with powdery mildew.\n"
        f"* 2 - The client is interested in predicting if a cherry leaf is "
        f"healthy or contains powdery mildew."
        )
