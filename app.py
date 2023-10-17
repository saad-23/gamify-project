from flask import Flask, request, jsonify
from flask_cors import CORS
import synonyms_get
import re



#nlp = spacy.load("en_core_web_md")

app= Flask (__name__)
CORS(app)
#nlp = spacy.load('en_core_web_sm')

@app.route('/')
def index():
    return 'Hello World'

@app.route('/<name>')
def print_name(name):
    return 'Hi ,{}'.format(name)

@app.route('/synonyms/<string:fr>',methods=['GET','POST'])
def synonyms(fr):
    if request.method == 'GET':
        if len(fr) > 0 :
            syns1 = synonyms_get.create_synonym_dict()
            
            # Example usage
            #original_sentence = "The doll was found in garage. The toy could be sold now. The doll was replaced with a dog. "
            #original_sentence ="The software will be used by user to print account report. The report will be printable by employees through system."
            #original_sentence = 'run race stroll rush nice lovely mean kind'
            #original_sentence = "Only collision estimators shall search for recycled parts.  Users without the collision estimator role shall not have access to the search for recycled parts."
            #original_sentence = "Movies can only be streamed if the customer has purchased a movie  and is within the 2-day time period to stream the movie.Daily usage statistics should be logged  and accessible by the user."
            #original_sentence = "The product will require collaboration with a database management system (DBMS).The DBMS may be located on the same machine as the product or on a separate machine residing on the same computer network."
            original_sentence = fr
            fr = None
            new_sentence = synonyms_get.replace_synonyms_in_sentence(original_sentence,syns1)
            #print("Original Sentence:", original_sentence)
            new_sentence= sorted(new_sentence, key=lambda x: x[0])
            #print("Modified Sentence:", new_sentence )

            return jsonify(new_sentence)
            #return jsonify(syns)
        else:
            return 'Nothing Found',404
#if __name__ == '__main__':
#    app.run()
