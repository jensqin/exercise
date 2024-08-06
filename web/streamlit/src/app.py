import streamlit as st
import polars as pl

st.write("Here's our first attempt at using data to create a table:")
st.write(
    pl.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
)
