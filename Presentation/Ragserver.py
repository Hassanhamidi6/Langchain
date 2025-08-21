from flask import Flask, request, jsonify, render_template
from Rag import load_vector_store, create_vector_store, get_response


create_vector_store("NIPS-2017-attention-is-all-you-need-Paper.pdf", "fitness_trainer")
retriever=load_vector_store("fitness")

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    query=request.get_json().get("query")
    response=get_response(retriever,query)
    return jsonify({"response": response})





if __name__ =="__main__":
    app.run(debug=True)


# name = "taha" 
# print(name)

# p1= Person(age=25 name="Taha")