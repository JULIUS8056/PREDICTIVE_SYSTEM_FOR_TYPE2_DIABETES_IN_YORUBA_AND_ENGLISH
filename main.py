import streamlit as st
st.set_page_config(
    page_title="Multiple App",
    layout="wide"
)
with open("design.css") as ds:
    st.markdown(f"<style>{ds.read()}</style>", unsafe_allow_html=True)
st.title("Clinical Decision Support System for early detection of type 2 diabetes ".upper())
st.sidebar.success("Select your preffered Language above.")
