from flask import Flask,render_template,request
app = Flask(__name__)
import pickle



file = open('model.pkl','rb')
clf= pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myOict =request.form
        Fever = int(myOict['Fever'])
        age = int(myOict['age'])
        pain = int(myOict['pain'])
        runnynose = int(myOict['runnynose'])
        difficultyforbreath = int(myOict['difficultyforbreath'])

        inputFeatures = [Fever, pain, age, runnynose, difficultyforBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
                #return 'hello,world'+ str(infProb)


    
if __name__ == "__main__":
    app.run(debug=True)    