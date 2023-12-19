import pandas as pd
from flask import Flask, request, jsonify


from pycaret.regression import load_model
from pycaret.regression import predict_model


app = Flask(__name__)
model= load_model("../../models/model_v1")


@app.route('/predictOne', methods =['POST'])
def predictOne():
    data=request.json
    data_to_predict = pd.json_normalize(data)
    try:
        prediccion = predict_model(model, data=data_to_predict)
        valor_predicho = round(list(prediccion['prediction_label'])[0],4)
        print(valor_predicho)
        return jsonify({'Prediccion':valor_predicho})
    except:
        return jsonify({'mensaje': "Se generó un error en la predicción."})


model2= load_model("../../models/model_v2")

@app.route('/predictTwo', methods =['POST'])
def predictTwo():
    data=request.json
    data_to_predict = pd.json_normalize(data)
    try:
        prediccion = predict_model(model2, data=data_to_predict)
        valor_predicho = round(list(prediccion['prediction_label'])[0],4)
        print(valor_predicho)
        return jsonify({'Prediccion':valor_predicho})
    except:
        return jsonify({'mensaje': "Se generó un error en la predicción."})

model3= load_model("../../models/model_v3")

@app.route('/predictThree', methods =['POST'])
def predictThree():
    data=request.json
    data_to_predict = pd.json_normalize(data)
    try:
        prediccion = predict_model(model3, data=data_to_predict)
        valor_predicho = round(list(prediccion['prediction_label'])[0],4)
        print(valor_predicho)
        return jsonify({'Prediccion':valor_predicho})
    except:
        return jsonify({'mensaje': "Se generó un error en la predicción."})


@app.route('/saludo',methods=['GET'])
def saludo():
    strOut = "hola mundo"
    print(strOut)
    return jsonify({'mensaje':strOut})

#@app.route('/sumar/<int:a>/<int:b>',methods=['GET'])
#def sumar(a, b):
#    resultado = a + b
#    return jsonify({'suma':resultado})

@app.route('/sumar/', methods=['GET'])
@app.route('/sumar/<int:a>/<int:b>', methods=['GET'])
def sumar(a=None, b=None):
    if((a==None) and (b==None)):
        return jsonify({'resultado': 'No se enviaron parámetros para operar.'})
    else:
        resultado = a+b
        return jsonify({'resultado': resultado})
    
@app.route('/mul/',  methods=['GET'])
def multiplicar():
    try:
        a = request.args.get('a', None)
        b = request.args.get('b', None)
        resultado = a * b
        return jsonify({'reultado': resultado})
    except:
        return jsonify({'resultado': 'No se enviaron parámetros para operar.'})

@app.route('/div', methods =['POST'])
def division():
    data=request.json
    print(data)
    return jsonify({'mensaje':'ok'})

