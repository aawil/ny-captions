from flask import Flask, request, render_template, jsonify
from textgenrnn import textgenrnn
from keras import backend as K

# Initialize the app

app = Flask(__name__)

flag = True
tg = None
cat_flag = None

@app.route("/")
def hello():

    return render_template("index.html")

@app.route("/generate")
def generate():
    global flag
    global tg
    global cat_flag

    cat = request.args.get('cat')
    cat = cat.lower()
    if cat=="anything else":
        cat = "other"

    if flag:
        tg = textgenrnn(name=cat)
        tg.load(f"./static/weights/{cat}_weights.hdf5")
        flag = False
        cat_flag = cat
    elif cat_flag!=cat:
        K.clear_session()
        tg = textgenrnn(name=cat)
        tg.load(f"./static/weights/{cat}_weights.hdf5")
        cat_flag = cat
    else:
        pass
    
    caption = tg.generate(return_as_list=True, temperature=0.5)[0]

    return jsonify(result=caption)


if __name__=="__main__":
    # For local development:
    app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    #app.run()
