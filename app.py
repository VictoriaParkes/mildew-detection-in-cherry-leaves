import streamlit as st
from app_pages.multi_page import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_data_visualization import page_data_visualization_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_model_performance import page_model_performance_body

# Create an instance of the app
app = MultiPage(app_name="Powdery Mildew Detector")

# Add app pages
app.add_page("Project Summary", page_summary_body)
app.add_page("Leaf Visualization Study", page_data_visualization_body)
app.add_page("Powdery Mildew Detection", page_mildew_detector_body)
app.add_page("Project Hypotheses", page_project_hypothesis_body)
app.add_page("Model Performance", page_model_performance_body)

app.run()