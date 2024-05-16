import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
st.title("Attendance for Day {}".format(date))

count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")
if count == 0:
    st.write("Count is zero")
elif count % 15 == 0:
    st.write("FizzBuzz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write("Count: ", count)

df = pd.read_csv("/Users/khatuaryan/Desktop/Aryan/Studies/Projects/Python/FaceRecognition/Attendance/Attendance_" + date + ".csv")
st.dataframe(df.style)
