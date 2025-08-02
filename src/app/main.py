"""Streamlit entry point for the AIOCR application.

This module sets up simple navigation and redirects to the main OCR page.
"""

import streamlit as st


def main() -> None:
    """Redirect to the main OCR page while exposing navigation links."""
    st.set_page_config(page_title="AIOCR")

    st.sidebar.page_link("1_Main_OCR.py", label="Main OCR")
    st.sidebar.page_link("pages/1_Review.py", label="Review")
    st.sidebar.page_link("pages/2_Dashboard.py", label="Dashboard")

    st.switch_page("1_Main_OCR.py")


if __name__ == "__main__":
    main()

