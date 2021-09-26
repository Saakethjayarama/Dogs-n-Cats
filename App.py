from flask import Flask, render_template, request
import os
from model import predict

app = Flask(__name__)


@app.route('/',methods=['GET', 'POST'])

def getIndex():
  if request.method == 'POST':
    f = request.files['file']
    f.save(f.filename)
    res = predict('./{}'.format(f.filename))
    os.remove(f.filename)
    return render_template('index.html', prediction=res)
  else: 
    return render_template('index.html')

 

app.run(debug=True)