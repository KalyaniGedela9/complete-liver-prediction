import os
from flask import Flask,render_template,request,redirect, url_for, session
from flask_mysqldb import MySQL
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import re
import MySQLdb.cursors
from sklearn.preprocessing import StandardScaler
from flask_mysqldb import MySQL
app = Flask(__name__)
app.secret_key = "super secret key"
model=pickle.load(open('logic.pkl' , 'rb'))
app.config['MYSQL_HOST']= "localhost"
app.config['MYSQL_USER']= "root"
app.config['MYSQL_PASSWORD']="rootpassword"
app.config['MYSQL_DB']="liverdata"
app.config['MYSQL_CURSORCLASS']='DictCursor'
mysql=MySQL(app)
@app.route('/')
#@app.route('/')
def stocks():
    filename = 'liver1.csv'
    data = pandas.read_csv(filename, header=0)
    stocklist = list(data.values)
    print(data['Dataset'])
    print(data.head())
    print(data.info())
    print("we can take 2 as 0 and 1 as 1")
    data['Dataset'] = data['Dataset'].map({2: 0, 1: 1})
    print(data['Dataset'].value_counts())
    data['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True)
    data_features = data.drop(['Dataset'], axis=1)
    data_num_features = data.drop(['Gender', 'Dataset'], axis=1)
    print(data_num_features.head())
    print(data_num_features.describe())
    # plt.figure(figsize=(8, 6))
    # data.groupby('Gender').sum()['Total_Bilirubin'].plot.bar(color='fuchsia')
    print("data was preprocessing")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cols = list(data_num_features.columns)
    data_features_scaled = pandas.DataFrame(data=data_features)
    data_features_scaled[cols] = scaler.fit_transform(data_features[cols])
    print(data_features_scaled.head())
    print("Create dummies for our data")
    data_exp = pandas.get_dummies(data_features_scaled)
    print(data_exp.head())
    print("dived the data into test and train modules")
    X = data_exp[['Age', 'Gender_Female', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                  'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                  'Albumin_and_Globulin_Ratio']]
    y = data['Dataset']
    print(X)
    print(y)
    print("#we split the training and testing in a certain ratio as 80% training and 20% for testing")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    score = logmodel.score(X_test, y_test)
    print("prediction score is(Accuracy):", score * 100)
    return render_template('stocks.html',stocklist=stocklist)
@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/Home')
def Home():
    return render_template('admin.html')
@app.route('/patient1',methods=['GET','POST'])
def patient1():
    if request.method=='POST':
        pati=request.form
        name=pati['Pname']
        contact=pati['Contact']
        Date=pati['DOB']
        Blood=pati['Blood Grp']
        cur = mysql.connection.cursor()
        #cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        #Patient = cursor.fetchall()
        cur.execute("INSERT INTO Patient(name,contact,Date,Blood) values (%s,%s,%s,%s)",(name,contact,Date,Blood))
        mysql.connection.commit()
        cur.close()
    return render_template('front1.html')
@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM validate WHERE username = % s AND password = % s', (username, password, ))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            msg = ''
            return render_template('patient.html', msg = msg)
        else:
            msg='Please enter valid username and password'
    return render_template('login.html', msg = msg)
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM validate WHERE username = % s', (username, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'please fill the form !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO validate VALUES (NULL, % s, % s, % s)', (username, password, email, ))
            mysql.connection.commit()
            msg = ''
            return render_template('patient.html',msg=msg)
    elif request.method == 'POST':
        msg='please fill the details'
    return render_template('register.html', msg = msg)
@app.route('/front',methods=['GET','POST'])
def front():
    if request.method=='POST':
        user=request.form
        Age=user['Age']
        Gender=user['Gender_Female']
        Total_bilirubin=user['Bilirubin']
        Direct_bilirubin=user['DBilirubin']
        A_Phosphotase=user['Alkalin']
        Al_Aminotransferase=user['Alamine']
        Asparatate=user['Asparatate']
        Total_proteins=user['Proteins']
        Albumin=user['Albumin']
        A_Globulin=user['AGRatio']
        cur=mysql.connection.cursor()
        cur.execute("INSERT INTO liver(Age,Gender,Total_bilirubin,Direct_bilirubin,A_Phosphotase, Al_Aminotransferase,Asparatate,Total_proteins,Albumin,A_Globulin) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(Age,Gender,Total_bilirubin,Direct_bilirubin,A_Phosphotase, Al_Aminotransferase,Asparatate,Total_proteins,Albumin,A_Globulin))
        mysql.connection.commit()
        cur.close()
        return predict()
    return render_template('front.html')
standard_to=StandardScaler()
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Age=int(request.form['Age'])
        Gender_Female=request.form['Gender_Female']
        if(Gender_Female=='Female'):
            Gender_Female=1
            Gender_Male=0
        else:
            Gender_Female=0
            Gender_Male=1
        Total_bilirubin=float(request.form['Bilirubin'])
        Direct_bilirubin=float(request.form['DBilirubin'])
        A_Phosphotase=float(request.form['Alkalin'])
        Al_Aminotransferase=float(request.form['Alamine'])
        Asparatate=float(request.form['Asparatate'])
        Total_proteins=float(request.form['Proteins'])
        Albumin=float(request.form['Albumin'])
        A_Globulin=float(request.form['AGRatio'])
        prediction=model.predict([[Age,Gender_Female,Total_bilirubin,Direct_bilirubin,A_Phosphotase,Al_Aminotransferase,Asparatate,Total_proteins,Albumin,A_Globulin]])
        output=round(prediction[0],2)
        if output<1:
            print("probability of chance:",output)
            return render_template('positive.html',prediction_text="their is chance of getting liver disease")
        else:
            return render_template('high.html',prediction_text="their is chance of getting liver disease")
    else:
        return render_template('front.html')
@app.route('/front1',methods=['GET','POST'])
def front1():
    if request.method=='POST':
        user=request.form
        Date=user['DOB']
        Age=user['Age']
        Gender=user['Gender_Female']
        Total_bilirubin=user['Bilirubin']
        Direct_bilirubin=user['DBilirubin']
        A_Phosphotase=user['Alkalin']
        Al_Aminotransferase=user['Alamine']
        Asparatate=user['Asparatate']
        Total_proteins=user['Proteins']
        Albumin=user['Albumin']
        A_Globulin=user['AGRatio']
        cur=mysql.connection.cursor()
        cur.execute("INSERT INTO Uliver2(Date,Age,Gender,Total_bilirubin,Direct_bilirubin,A_Phosphotase, Al_Aminotransferase,Asparatate,Total_proteins,Albumin,A_Globulin) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(Date,Age,Gender,Total_bilirubin,Direct_bilirubin,A_Phosphotase, Al_Aminotransferase,Asparatate,Total_proteins,Albumin,A_Globulin))
        mysql.connection.commit()
        cur.close()
        return predict1()
    return render_template('front1.html')
standard_to=StandardScaler()
@app.route("/predict1",methods=['POST'])
def predict1():
    if request.method == 'POST':
        Age=int(request.form['Age'])
        Gender_Female=request.form['Gender_Female']
        if(Gender_Female=='Female'):
            Gender_Female=1
            Gender_Male=0
        else:
            Gender_Female=0
            Gender_Male=1
        Total_bilirubin=float(request.form['Bilirubin'])
        Direct_bilirubin=float(request.form['DBilirubin'])
        A_Phosphotase=float(request.form['Alkalin'])
        Al_Aminotransferase=float(request.form['Alamine'])
        Asparatate=float(request.form['Asparatate'])
        Total_proteins=float(request.form['Proteins'])
        Albumin=float(request.form['Albumin'])
        A_Globulin=float(request.form['AGRatio'])
        prediction=model.predict([[Age,Gender_Female,Total_bilirubin,Direct_bilirubin,A_Phosphotase,Al_Aminotransferase,Asparatate,Total_proteins,Albumin,A_Globulin]])
        output=round(prediction[0],2)
        if output<1:
            print("probability of chance:",output)
            return render_template('positive1.html',prediction_text="their is chance of getting liver disease")
        else:
            return render_template('high1.html',prediction_text="their is chance of getting liver disease")
    else:
        return render_template('front1.html')
@app.route('/patient',methods=['GET','POST'])
def patient():
    if request.method=='POST':
        pati=request.form
        name=pati['Pname']
        contact=pati['Contact']
        Date=pati['DOB']
        Blood=pati['Blood Grp']
        cur = mysql.connection.cursor()
        #cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        #Patient = cursor.fetchall()
        if name==" ":
            msg="plz fill form"
        else:
            cur.execute("INSERT INTO Patient(name,contact,Date,Blood) values (%s,%s,%s,%s)",(name,contact,Date,Blood))
        mysql.connection.commit()
        cur.close()
    return render_template('front.html')
@app.route('/patient2')
def patient2():
    return render_template('patient1.html')
'''@app.route('/new')
def new():
    filename = 'liver1.csv'
    data = pandas.read_csv(filename, header=0)
    stocklist = list(data.values)
    
    return render_template('stocks.html', stocklist=stocklist)'''
@app.route('/data')
def data():
    return render_template('fr.html')
if __name__=="__main__":
    app.run(debug=True)
    
