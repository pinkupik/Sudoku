from app.src import filter
from app.gui import kuskus
from app import kokos
from app.gui import display
import streamlit as st

def main():
    st.set_page_config(layout="wide")  # Use the whole width of the screen
    display.display()

if __name__ == "__main__":
    main()