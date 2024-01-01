import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics


from web_function import train_model

def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('Visualisasi Prediksi kesetiaan Nasabah')

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x,y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        y_pred = model.predict(x_test)
        labels = ['0', '1']
        confusmat = confusion_matrix(y_test, y_pred)
        dis = metrics.ConfusionMatrixDisplay(confusion_matrix=confusmat, display_labels=labels)
        dis.plot()
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x,y)
        plt.figure(figsize=(30,10))
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=4, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['0', '1']
        )
        st.graphviz_chart(dot_data)
    