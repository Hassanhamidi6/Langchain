from flask import Flask, request, jsonify, render_template
from RagByKrish import load_vector_store, get_response, create_vector_store

# Step 1: Create vector store from PDF
load_vector_store("Large_Medical_Information_Handbook.pdf", "Health")  # Creates and saves the vector store in a folder

# Step 2: Load retriever from the saved vector store
retriever = create_vector_store("Medical_assistant")   # Now loads from the correct folder

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/gnerate_resposne", methods=["POST"])
def generate_response():
    query=request.get_json().get("query")
    response=get_response(retriever, query)
    return jsonify({"response": str(response.content)})



if __name__ =="__main__":
    app.run(debug=True)