

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('final qda model')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('irispred.jpg')
    image_office = Image.open('iris.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict the class of iris plant.')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Predicting class of iris plant")
    if add_selectbox == 'Online':
        sepal_length=st.number_input('sepal length in cm' , min_value=0.1, max_value=7.9, value=0.1)
        sepal_width=st.number_input('sepal width in cm',min_value=0.1, max_value=4.4, value=0.1)
        petal_length = st.number_input('petal length in cm', min_value=0.1, max_value=6.9, value=0.1)
        petal_width= st.number_input('petal width in cm', min_value=0.1, max_value=2.5, value=0.1)
        output=""
        input_dict={'sepal_length':sepal_length,'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
