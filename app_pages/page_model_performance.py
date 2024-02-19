import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_model_performance_body():
    st.write("### Model Performance")

    version = "v2"

    st.write("### Train, Validation and Test Set: Labels Frequencies")
    st.write(f"The data was split into 3 sub-datasets to help prevent over or "
             f"underfitting during ML model training, as follows:")

    labels_distribution_bar = plt.imread(
        f"outputs/{version}/labels_distribution_bar.png"
    )
    st.image(
        labels_distribution_bar,
        caption='Labels Distribution on Train, Validation and Test Sets'
    )
    labels_distribution_pie = plt.imread(
        f"outputs/{version}/data_distribution_pie.png"
    )
    st.image(
        labels_distribution_pie,
        caption='Labels Distribution on Train, Validation and Test Sets'
    )
    st.write("---")

    st.write("### Model Training History")
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write(f"The model learning curves suggest the model learned well as "
             f"both the loss and accuracy plots for training and validation "
             f"data follow a similar path and are close to each other.")
    st.write("---")

    st.write("### Confusion Matrix")

    confusion_matrix = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix, caption="Confusion Matrix")
    st.write(f"The confusion matrix plot shows that when testing the model "
             f"with the test set; 419 healthy leaves were correctly predicted "
             f"to be healthy, 3 healthy leaves were incorrectly predicted to "
             f"be mildew infected, and all 422 mildew infected leaves were "
             f"correctly predicted to be mildew infected, no mildew infected "
             f"leaves were incorrectly predicted to be healthy.")
    st.write("---")

    st.write("### Classification Report")

    classification_report = plt.imread(
        f"outputs/{version}/classification_report.png"
    )
    st.image(classification_report, caption="Classification Report")
    st.write(f"Recall/sensitivity rate is the percentage of the class that "
             f"was properly predicted. The classification report shows that "
             f"99.3% of the healthy leaf images were correctly predicted as "
             f"healthy and 100% of the mildew infected leaf images were "
             f"correctly predicted as infected.\n\n"
             f"Precision is the percentage of predictions related to a class "
             f"made were correct, or how many predictions of a certain class "
             f"were correct compared to the total number of predictions of "
             f"that class. The classification report shows that the 100% of "
             f"the healthy class predictions made were correct and 99.3% of "
             f"the powdery_mildew class predictions were correct.\n\n"
             f"The f1-score measures Recall and Precision together using "
             f"Harmonic Mean. It gives the average value for Recall and "
             f"Precision.")
    st.write("---")

    st.write("### AUC - ROC Curve")

    roc_curve = plt.imread(f"outputs/{version}/roc_curve.png")
    st.image(roc_curve, caption="AUC - ROC Curve")
    st.write(f"ROC (Receiver Operating Characteristic) is a probability curve "
             f"and is used to calculate the AUC (Area Under Curve) value. The "
             f"AUC value represents the degree or measure of separability, "
             f"which is the models capability to distinguish between classes. "
             f"The AUC value achieved in this evaluation report shows that "
             f"the model has a high capability to distinguish between "
             f"classes.")
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(
        pd.DataFrame(
            load_test_evaluation(version),
            index=["Loss", "Accuracy"]
        )
    )
    st.write(f"The evaluation of the model over the data test set gave a "
             f"generalized loss of less than 1% and accuracy of more than "
             f"99%, which more than satisfies the clients requirement of 97% "
             f"accuracy.")
