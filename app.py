import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import time

# Ícone e nome do app
st.set_page_config(
    page_title="Previsão Titanic",  # Nome que aparecerá na guia
    page_icon="🚢",  # Pode ser um emoji ou o caminho para um arquivo de imagem (como .png ou .ico)
)

# Título do App
st.image("./resources/titanic.png")
st.title('FdsfsdiufuhNR - Previsão de Sobrevivência no Titanic')

# Carregar o modelo treinado
with open('modelo_titanic.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a dialog function
@st.dialog("Result")
def modal_dialog(fig, prediction_proba, message):

    gauge_placeholder = st.empty()
    gauge_placeholder.plotly_chart(fig)

    # Animação de "carregamento"
    for i in range(0, int(prediction_proba * 100) + 1, 5):  # Atualiza de 5 em 5
        fig.update_traces(value=i)
        gauge_placeholder.plotly_chart(fig)
        time.sleep(0.05)  # Pausa para criar o efeito de animação

    # Ajustar o valor final para ser exato ao final da animação
    fig.update_traces(value=prediction_proba * 100)
    gauge_placeholder.plotly_chart(fig)

    if (prediction_proba < 0.4):
        st.error(message)
    elif (prediction_proba < 0.6):
        st.warning(message)
    else:
        # st.balloons()
        st.success(message)

# Formulário para entrada de dados do usuário
with st.form(key='formulario_previsao'):
    st.subheader('Insira os dados do passageiro para previsão')

    # pclass = st.selectbox('Classe (1 = Primeira, 2 = Segunda, 3 = Terceira)', [1, 2, 3], key='pclass')
    pclass_option = st.selectbox('Classe', ["Primeira", "Segunda", "Terceira"], key='pclass')
    if (pclass_option == "Primeira"):
        pclass = 1
    elif (pclass_option == "Segunda"):
        pclass = 2
    else:
        pclass = 3
        
    sex_option = st.radio("Sexo", ["Masculino", "Feminino"])
    sex = 0 if sex_option == "Masculino" else 1 
    age = st.number_input("Idade", 0, 100)
    fare = st.slider('Tarifa (Fare)', 0.0, 500.0, 35.0, key='fare')
    embarked_option = st.selectbox('Porto de Embarque', ["C", "Q", "S"], key='embarked')
    if (embarked_option == "C"):
        embarked = 0
    elif (embarked_option == "Q"):
        embarked = 1
    else:
        embarked = 2

    # Botão dentro do formulário
    submit_button = st.form_submit_button(label='Verificar Sobrevivência')

# Quando o botão do formulário for clicado, faz a previsão, versão completa
if submit_button:
    input_data = np.array([[pclass, sex, age, fare, embarked]])

    # with st.spinner(text="In progress"):
    #     time.sleep(3)
    
    # prediction = model.predict(input_data)
    # result = 'Sobreviveu' if prediction[0] == 1 else 'Não Sobreviveu'

    # Exibir o resultado da previsão
    # st.subheader(f'Resultado: {result}')
    
    # Obter as probabilidades de sobrevivência
    prediction_proba = model.predict_proba(input_data)[0][1]  # Pega a probabilidade de "sobreviveu"
    st.subheader(f'Probabilidade de Sobrevivência: {prediction_proba * 100:.2f}%')

    if (prediction_proba < 0.4):
        message = "Infelizmente suas changes não são boas companheiro, espero que saiba nadar e aguente o frio!🥶"
    elif (prediction_proba < 0.6):
        message = "O futuro é incerto para você, desejo boa sorte na sua fuga!😐"
    else:
        message = "As chances estão do seu lado, parece que consiguirá sobreviver no fim, que bom!😁"

    # Gráfico de velocímetro (gauge chart) usando Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,#prediction_proba * 100,  # Mostrar a probabilidade em porcentagem
        title={'text': "Probabilidade de Sobrevivência"},
        gauge={
            'bar': {'color': "darkblue"},
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "lightgreen"}
            ]
        }
    ))

    # Exibir o gráfico no Streamlit
    gauge_placeholder = modal_dialog(fig, prediction_proba, message)
    fig.update_traces(value=prediction_proba * 100)
    st.plotly_chart(fig)
    
    if (prediction_proba < 0.4):
        st.error(message)
    elif (prediction_proba < 0.6):
        st.warning(message)
    else:
        st.balloons()
        st.success(message)
