import streamlit as st

from web_function import predict

def app(df, x, y):
    st.title("Halaman Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        CreditScore	= st.text_input("Input nilai CreditScore")
    with col1:
        Geography = st.selectbox("Geography (Keterangan: 0 = France, 1 = Germany, 2 = Spain)", ["0", "1", "2"])
    with col1:
        Gender = st.selectbox("Input Gender (Keterangan: 0 = Female, 1 = Male)", ["0", "1"])
    with col1:
        Age = st.text_input("Input Usia")
    with col1:
        Tenure = st.select_slider("Tenure", ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    with col2:
        Balance = st.text_input("Input nilai Balance")
    with col2:
        NumOfProducts = st.select_slider("Num of products", ['1','2','3','4'])
    with col2:
        HasCrCard = st.selectbox("Apakah orang tersebut memiliki kartu kredit (Keterangan: 0 = No, 1 = Yes)", ["0", "1"])
    with col2:
        IsActiveMember = st.selectbox("Apakah orang tersebut member aktif (Keterangan: 0 = No, 1 = Yes)", ["0", "1"])
    with col2:
        EstimatedSalary = st.text_input("Input nilai Estimated Salary")

    features = [CreditScore, Geography,	Gender,	Age, Tenure, Balance, NumOfProducts, HasCrCard,	IsActiveMember,	EstimatedSalary]

    if st.button("Prediksi"):
        prediction, score = predict(x,y,features)
        score=score
        st.info("Prediksi Berhasil")

        if(prediction==1):
            st.warning("Nasabah meninggalkan/beralih dari bank tersebut")
        else:
            st.success("Nasabah tetap memilih bank tersebut")
        
        st.write("Model yang digunakan memiliki tingkat akurasi ", (score*100), "%")
