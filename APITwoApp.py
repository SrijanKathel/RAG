from flask import jsonify,request,Flask
from APITwoMain import query_rag
from APITwoVector import load_documents
import os

app = Flask(__name__)


@app.route("/hackrx/run", methods = ["POST"])
def process_json():
    data = request.get_json()
    # doc_url = data.get("documents"," ")
    doc_url = data.get("documents")
    # questions = data.get("questions",[])
    questions = data.get("questions")
    answers = []
    
    if (not data) or ("questions" not in data) or ("documents" not in data): 
        return jsonify({"Error":"enter complete JSON"}) , 400
    
    else:
        #handling doc,db and vector store part
        
        local_url = download_doc(doc_url)
        load_documents(local_url)
        
        #handling question and response part
        
        for question in questions:
            response = query_rag(question)
            answers.append(response)
            
        return jsonify({"answers":answers})
    

def download_doc(doc_url,directory = "data" , filename = "dataDoc.pdf"):
    os.makedirs(directory, exist_ok=True)
    local_url = os.path.join(directory, filename)
    return local_url


if __name__ == "__main__":
    app.run(host="0.0.0.0" , port = 5000, debug=True)



    
    
    
    
    
    