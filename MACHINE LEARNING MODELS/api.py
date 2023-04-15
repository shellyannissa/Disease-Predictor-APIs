from flask import Flask,request,jsonify
import pickle
import numpy as np
import json

model=pickle.load(open('symptoms.pkl','rb'))
model_diab=pickle.load(open('diab.pkl','rb'))
model_cvd=pickle.load(open('cvd.pkl','rb'))
model_chr_kid=pickle.load(open('chr_kid_dis.pkl','rb'))
model_stroke=pickle.load(open('stroke.pkl','rb'))
model_thyroid=pickle.load(open('thyroid.pkl','rb'))
model_parkinsons=pickle.load(open('parkinsons.pkl','rb'))
model_lung=pickle.load(open('lung_cancer.pkl','rb'))
model_cerv=pickle.load(open('cerv_canc.pkl','rb'))
model_age_chr_kid=pickle.load(open('age_chr_kid_dis.pkl','rb'))
model_age_cvd=pickle.load(open('age_cvd.pkl','rb'))
model_age_diab=pickle.load(open('age_diab.pkl','rb'))
model_age_stroke=pickle.load(open('age_stroke.pkl','rb'))
model_age_thyroid=pickle.load(open('age_thyroid.pkl','rb'))
model_age_lung=pickle.load(open('age_lung_cancer.pkl','rb'))
model_age_cerv=pickle.load(open('age_cerv_canc.pkl','rb'))
app=Flask(__name__)



@app.route('/symp',methods=['POST'])
def symp():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    dis = model.predict(arr)[0]
    result = {'result': dis}
    # return the result as a JSON object
    return jsonify(result)


@app.route('/diab',methods=['POST'])
def diab():
    data = request.json['data']

    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_diab.predict(arr)[0]
    accuracy_diab=81
    result= {'result': int(num),'acc':accuracy_diab}
    # return the result as a JSON object
    return jsonify(result)



@app.route('/cvd',methods=['POST'])
def cvd():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_cvd.predict(arr)[0]
    accuracy_cvd=74.7
    result= {'result': int(num),'acc':accuracy_cvd}
    # return the result as a JSON object
    return jsonify(result)
    

@app.route('/chr_kid',methods=['POST'])
def chr_kid():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_chr_kid.predict(arr)[0]
    accuracy_chr_kid=97.5
    result= {'result': int(num),'acc':accuracy_chr_kid}
    # return the result as a JSON object
    return jsonify(result)


@app.route('/stroke',methods=['POST'])
def stroke():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_stroke.predict(arr)[0]
    accuracy_stroke=95.6
    result= {'result': int(num),'acc':accuracy_stroke}
    # return the result as a JSON object
    return jsonify(result)
  

@app.route('/lung',methods=['POST'])
def lung():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_lung.predict(arr)[0]
    accuracy=96.7
    result= {'result': int(num),'acc':accuracy}
    # return the result as a JSON object
    return jsonify(result)  
    

@app.route('/cerv_canc',methods=['POST'])
def cerv_canc():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_cerv.predict(arr)[0]
    accuracy=98.8
    result= {'result': int(num),'acc':accuracy}
    # return the result as a JSON object
    return jsonify(result) 


@app.route('/thyroid',methods=['POST'])
def thyroid():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_thyroid.predict(arr)[0]
    accuracy_thyroid=99.7
    result= {'result': int(num),'acc':accuracy_thyroid}
    # return the result as a JSON object
    return jsonify(result)
    
    
    
@app.route('/parkinsons',methods=['POST'])
def parkinsons():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_parkinsons.predict(arr)[0]
    accuracy_parkinsons=92.3
    result= {'result': int(num),'acc':accuracy_parkinsons}
    # return the result as a JSON object
    return jsonify(result)
    
    
@app.route('/age_chr_kid',methods=['POST'])
def age_chr_kid():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_age_chr_kid.predict(arr)[0]
    result = {'result': int(num)}
    # return the result as a JSON object
    return jsonify(result)
    
  
@app.route('/age_cvd',methods=['POST'])
def age_cvd():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_age_cvd.predict(arr)[0]
    result = {'result': int(num)}
    # return the result as a JSON object
    return jsonify(result)
    
    
@app.route('/age_diab',methods=['POST'])
def age_diab():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_age_diab.predict(arr)[0]
    result = {'result': int(num)}
    # return the result as a JSON object
    return jsonify(result)
  
  
@app.route('/age_stroke',methods=['POST'])
def age_stroke():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_age_stroke.predict(arr)[0]
    result = {'result': int(num)}
    # return the result as a JSON object
    return jsonify(result)
    
    
@app.route('/age_thyroid',methods=['POST'])
def age_thyroid():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_age_thyroid.predict(arr)[0]
    result = {'result': int(num)}
    # return the result as a JSON object
    return jsonify(result)
    

@app.route('/age_lung',methods=['POST'])
def age_lung():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_age_lung.predict(arr)[0]
    result = {'result': int(num)}
    # return the result as a JSON object
    return jsonify(result)
        


@app.route('/age_cerv_canc',methods=['POST'])
def age_cerv_canc():
    data = request.json['data']
    # convert the array to a numpy array
    arr = np.array(data).reshape(1, -1)
    num = model_age_cerv_canc.predict(arr)[0]
    result = {'result': int(num)}
    # return the result as a JSON object
    return jsonify(result)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)
    
    
    
    
