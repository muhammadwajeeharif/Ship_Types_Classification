import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image,ImageOps

image = Image.open(r"ship.jpg")
st.title("**Ship Type Classification:**")
st.image(image,use_column_width="auto")
st.header("About the Data:")
st.markdown("Ship or vessel detection has a wide range of applications, in the areas of maritime safety, fisheries management, marine pollution, defence and maritime security, protection from piracy, illegal migration, etc.")
st.markdown("Keeping this in mind, a Governmental Maritime and Coastguard Agency is planning to deploy a computer vision based automated system to identify ship type only from the images taken by the survey boats. You have been hired as a consultant to build an efficient model for this project.")
st.markdown('''There are 6252 images in train and 2680 images in test data. The five categories of ships are as follows:

1- Cargo Ship \n
2- Military Ship \n
3- Carrier Ship \n
4- Cruise Ship \n 
5- Tanker Ship ''')
upload_file = st.sidebar.file_uploader("Upload Ship Image", type = 'jpg')
generate_pred = st.sidebar.button("Predict")
st.text("                     ")

st.sidebar.markdown("Done by: **Muhammad Wajeeh Arif**")
st.sidebar.markdown("**Thankyou Baba G!**")
st.sidebar.markdown('<a href="https://www.linkedin.com/in/muhammad-wajeeh-arif-923b7917a/">Contact me via  LinkedIn !</a>', unsafe_allow_html=True)
model = tf.keras.models.load_model(r"https://l.facebook.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1D0cEm3wAtekgHQwB6qGMvXI6ojU2W3q4%2F%3Ffbclid%3DIwAR0apf4yvO9LkabPgxnHlUseENGHqOD29bfl29DDXN4pXtIv_0GH56BXY6k&h=AT01WtL99_iw6EgRog-FrVPuLTicYp881smaMDY4H0IIy4His0llTIse5e93FF_EsWZpY38PRBqCxccCu9rU55VfRxxqr6bgr4hRYsVRsXYW7E9Wx1J9YwwtSY0e24wlzGDq7Q")
def import_n_pred(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
if generate_pred:
    image = Image.open(upload_file)
    with st.expander('Ship Image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image, model)
    labels = ['Cargo Ship' , 'Military Ship', 'Carrier Ship', 'Cruise Ship', 'Tanker Ship']
    st.markdown("Prediction of Image is **{}**".format(labels[np.argmax(pred)]))
    st.success("Amazing!!!!")












