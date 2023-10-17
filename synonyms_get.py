import nltk
#nltk.download('averaged_perceptron_tagger')
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')
try:
  nltk.data.find('corpora/stopwords')
except LookupError:
  nltk.download('stopwords')
try:
  nltk.data.find('corpora/wordnet')
except LookupError:
  nltk.download('wordnet')

import requests
import en_core_web_md
nlp = en_core_web_md.load()

#nlp = spacy.load('en_core_web_sm')

from nltk.corpus import wordnet,stopwords
from nltk.tokenize import word_tokenize
#from autocorrect import Speller

#!python -m spacy download en_core_web_md


import asyncio
import aiohttp
import requests

from nltk.corpus import wordnet
event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

async def fetch_wordnet_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return synonyms

async def fetch_datamuse_synonyms(word):
    url = f"https://api.datamuse.com/words?rel_syn={word}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                synonyms = [item['word'] for item in await response.json()]
                return [syn for syn in synonyms if syn != word]
            else:
                return []

async def fetch_conceptnet_synonyms(word):
    base_url = 'https://api.conceptnet.io/query'
    word_encoded = requests.utils.quote(word)
    relationship = '/r/Synonym'
    query_url = f"{base_url}?rel={relationship}&start=/c/en/{word_encoded}"
    headers = {'Accept': 'application/json'}

    async with aiohttp.ClientSession() as session:
        async with session.get(query_url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                edges = data.get('edges', [])
                synonyms = [edge['end']['label'] for edge in edges if edge['end']['language'] == 'en']
                return [syn for syn in synonyms if syn != word]
            else:
                return []

async def fetch_all_synonyms(word, synonyms_dict1):
    tasks = [
        fetch_wordnet_synonyms(word),
        fetch_datamuse_synonyms(word),
        fetch_conceptnet_synonyms(word)
    ]

    syns = synonyms_dict1.get(word.lower(), [])
    tasks.append(asyncio.ensure_future(asyncio.sleep(0, result=syns)))

    results = await asyncio.gather(*tasks)
    return list(set(syn for sublist in results for syn in sublist))

def get_synonyms2(word, synonyms_dict1):
    try:
        return event_loop.run_until_complete(fetch_all_synonyms(word, synonyms_dict1))
    finally:
        pass  # Do not close the event loop here, as it's reused for multiple invocations


def get_synonyms(word,synonyms_dict1):
    synonyms = []

    '''for syn in wordnet.synsets(word):
      for lemma in syn.lemmas():
        if lemma.name() != word:
          #print(f"Wordnet: {lemma.name()}")
          synonyms.append(lemma.name())'''

    url = f"https://api.datamuse.com/words?rel_syn={word}"
    response = requests.get(url)
    if response.status_code == 200:
      synonyms__ = [item['word'] for item in response.json()]
      filtered_concepts = [syn for syn in synonyms__  if syn != word]
      #for syn in synonyms__:
          #if syn != word:
            #print(f"Datamuse: {syn}")
      synonyms.extend(filtered_concepts)
    
    #asyncio.run(main())

    #ConceptNet

    
    base_url = 'https://api.conceptnet.io/query'
    # Encode the word as a URL component
    word_encoded = requests.utils.quote(word)
   
    #### Synonymns
    relationship = '/r/Synonym'
    # Create the URL for the query
    query_url = f"{base_url}?rel={relationship}&start=/c/en/{word_encoded}"
    headers = {
        'Accept': 'application/json'
    }
    response = requests.get(query_url, headers=headers)
    isa_concepts =[]
    if response.status_code == 200:
        data = response.json()
        edges = data['edges']
        isa_concepts = [edge['end']['label'] for edge in edges if edge['end']['language'] == 'en']
        filtered_concepts = [syn for syn in isa_concepts if syn != word]
        #for syn in isa_concepts:
        # if syn != word:
            #print(f"ConceptNet:{syn}")
        synonyms.extend(filtered_concepts)

    # Custom Dictionry
    syns = synonyms_dict1.get(word.lower(), [])
    synonyms.extend(syns)
    #print(f'custom syn************************: {synonyms_dict1.get(word.lower(), [])}')
    #print(f'final : {word} : {synonyms}')
    return synonyms

def correct_spelling(text):
  spell = Speller()
  words = text.split()
  corrected_words = [spell(w) for w in words]
  return ' '.join(corrected_words)


def remove_stop_words(text):
  words = nltk.word_tokenize(text)
  stop_words = set(stopwords.words('english'))

  # Tokenize the text
  # Get the list of English stop words
  # Remove stop words from the tokenized words
  filtered_words = [word for word in words if word.lower() not in stop_words]
  # Join the filtered words to form a sentence
  filtered_text = ' '.join(filtered_words)
  return filtered_text


def create_synonym_dict():
    synonyms_dict1 = {
    "User": ["End user", "Client", "Customer", "Stakeholder","employee","buyer"],
    "Register": ["sign up"],
    "Login": ["Sign in"," log on"],
    "Log out": ["Sign out","Log off"],
    "System": ["Software", "Application", "Program", "Platform"],
    "Functionality": ["Features", "Capabilities", "Operations", "Functionalities"],
    "Use Case": ["Use case Scenario", "Application Scenario", "Usage Scenario"],
    "Requirement": ["Specification", "Condition", "Demand", "Prerequisite"],
    "UI": ["User Interface","GUI (Graphical User Interface)", "Front-end", "Interface Design"],
    "UX": ["User Experience","Usability", "User Satisfaction", "User Interaction"],
    "Input": ["Data Input", "User Input", "Information Entry","submit"],
    "Output": ["Results", "Outcome", "Generated Data"],
    "Authentication": ["Verification", "Login", "Access Control"],
    "Authorization": ["Permission", "Access Rights", "Clearance"],
    "Database": ["Data Repository", "Data Store", "Information Storage","db"],
    "Integration": ["Connectivity", "Interfacing", "System Integration"],
    "Security": ["Protection", "Safety", "Cybersecurity"],
    "Scalability": ["Expandability", "Growth Capacity", "Flexibility"],
    "Performance": ["Speed", "Efficiency", "Responsiveness"],
    "Compatibility": ["Interoperability", "Adaptability", "Cross-platform"],
    "Error Handling": ["Exception Handling", "Fault Tolerance", "Error Management"],
    "Logging": ["Event Logging", "Record Keeping", "Audit Trail"],
    "Log": ["record","history","audit"],
    "Documentation": ["User Manuals", "Guides", "Reference Materials"],
    "Testing": ["Quality Assurance", "Validation", "Verification"],
    "Test": ["Quality check","validate","verify"],
    "Deployment": ["Installation", "Rollout", "Release"],
    "Maintenance": ["Support", "Updates", "Upkeep"],
    "Version Control": ["Revision Control", "Source Code Management"],
    "Compliance": ["Adherence", "Conformity", "Regulatory Compliance"],
    "Backup and Recovery": ["Data Backup", "Disaster Recovery", "Data Restoration"],
    "Role": ["User role","User Types", "User Profiles", "Roles and Permissions"],
    "Workflow": ["Process Flow", "Task Flow", "Workflow Management"],
    "Reporting": ["Analytics", "Data Analysis", "Reporting Tools"],
    "Notifications": ["Alerts", "Messages", "Notifications System"],

    "Data": ["Information", "Data Elements", "Data Records","values"],
    "Algorithm": ["Procedure", "Method", "Logic"],
    "Validation": ["Data Validation", "Input Verification"],
    "validate":["verify"],
    "Output Generation": ["Result Presentation", "Output Display"],
    "User Authentication": ["User Verification", "User Login"],
    "User Authorization": ["Access Control", "Permissions Management"],
    "User Registration": ["User Signup", "Account Creation"],
    "User Profile": ["User Account", "User Information"],
    "Search Functionality": ["Search Features", "Search Capabilities"],
    "Reporting Module": ["Reporting System", "Report Generation"],
    "Audit Trail": ["Activity Log", "Logging History"],
    "Data Backup": ["Information Backup", "Data Preservation"],
    "Data Recovery": ["Information Restoration", "Data Retrieval"],
    "Feedback": ["User Input", "User Suggestions"],
    "Error": ["Error Notifications", "Exception"],
    "Software Architecture": ["System Design", "Application Structure"],
    "Documentation": ["User Guides", "Instruction Manuals"],
    "User Interface Design": ["UI Layout", "UI Styling","UI Design"],
    "Testing": ["Quality Testing", "Verification"],
    "Release Notes": ["Release Documentation", "Version Updates"],
    "Third-party Integration": ["External Services Integration", "API Integration"],
    "Role": ["Access Levels", "User Rights"],
    "Permission": ["Access Levels", "User Rights"],
    "Change": ["Software Updates", "Version Control"],
    "Hardware Requirements": ["System Hardware", "Equipment Specifications"],
    "Software Compatibility": ["Platform Compatibility", "System Support"],
    "Software Deployment": ["Application Installation", "Deployment Process"],
    "Training": ["Training Materials", "User Education"],
    "User Support": ["Customer Support", "Helpdesk Services"],
    "Data Privacy": ["Information Security", "Confidentiality Measures"],
    "Scalability": ["Expandability", "Growth Flexibility"],
    "Reliability": ["Stability", "Consistency"],
    "Usability": ["User-friendliness", "Ease of Use"],
    "Efficiency": ["Performance Optimization", "Speed"],
    "Customization": ["User Preferences", "Tailoring Options"],
    "Cross-platform Compatibility": ["Multi-platform Support", "Compatibility Across Systems"],

    }
    synonyms_dict1 = {key.lower(): [synonym.lower() for synonym in synonyms] for key, synonyms in synonyms_dict1.items()}
    return synonyms_dict1



#import torch
#from transformers import AutoModel, AutoTokenizer
#from scipy.spatial.distance import cosine

#model_name = "allenai/scibert_scivocab_uncased"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)


def calculate_glove_similarity(word1,word2):
    # Load the pre-trained GloVe model (you can specify a different dimension)
    # This will download the GloVe model if it's not already downloaded.
    # Define the two words you want to compare
    # Calculate the similarity between the two words
    doc1 = nlp(word1)
    doc2 = nlp(word2)

    if not doc1.has_vector or not doc2.has_vector:
        return 0.0  # Handle cases with empty vectors

    similarity = doc1.similarity(doc2)
    return similarity

import inflect

def plural_to_singular(plural_word):
    p = inflect.engine()
    singular_word = p.singular_noun(plural_word)
    return singular_word if singular_word else plural_word

import re
import time

import re

def remove_special_characters(input_string):
    # Define a regular expression pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'  # This pattern matches anything that is not a letter, digit, or whitespace

    # Use the sub() function to replace special characters with an empty string
    cleaned_string = re.sub(pattern, '', input_string)
    
    return cleaned_string

# Function to replace synonyms in a sentence
def replace_synonyms_in_sentence(sentence,syns1):
    
    

    stop_words = set(stopwords.words('english'))
    # Store calculated similarities in a dictionary for caching
    similarity_cache = {} # List to store suggestions
    suggestions = []

    tokens = word_tokenize((sentence))  # Tokenize the sentence
    filtered_tokens = [plural_to_singular(token.lower()) for token in tokens if remove_stop_words(token) != '' and remove_special_characters(token) != '']
    #filtered_tokens = [remove_special_characters(token) for token in tokens if remove_stop_words(token) != '']

    for token in filtered_tokens:
      #if remove_stop_words(token) != '' :
        #token = plural_to_singular(token)
        start_time = time.perf_counter()
        synonyms = get_synonyms2(token, syns1)
        synonyms = list(set(synonyms)) #unique synonyms
        #print(f'{token}:{synonyms}')
        end_time = time.perf_counter()# Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Time consumed by the function: {elapsed_time} seconds")
        #synonyms = get_synonyms2(token,syns1)  # Find synonyms for the token

        #print(f"checking {token}:{synonyms}")
        if synonyms:
            # Check if any synonym is already present in the sentence
            #if any(synonym in sentence for synonym in synonyms):
                #print(f" if synonyms: {synonyms}")
                # Pre-tokenize the sentence
                sentence_tokens = nltk.word_tokenize(sentence)
                sentence_tokens = [plural_to_singular(word.lower().replace("_", " ")) for word in sentence_tokens]
                synonyms = [plural_to_singular(word.lower().replace("_", " ")) for word in synonyms if word not in stop_words and remove_special_characters(word) != '']

                filtered_sentence_tokens = [(index, i) for index, i in enumerate(sentence_tokens) if i not in stop_words and remove_special_characters(i) != '' and i != token]

                  #output_string = re.sub(r'\b' + re.escape(word_to_find) + r'\b', replacement_word, input_string)
                  #sentence = re.sub(r'\b' + re.escape(word_to_find) + r'\b',token,sentence)
                for word_to_find in synonyms:
                  for index,i in filtered_sentence_tokens:
                      #if i not in stop_words and remove_special_characters(i) != '':
                            if (i, word_to_find) not in similarity_cache:
                                similarity = calculate_glove_similarity(i, word_to_find)
                                similarity_cache[(i, word_to_find)] = similarity
                            else:
                                similarity = similarity_cache[(i, word_to_find)]
                            #print(f"Similarity between '{i}' and '{word_to_find}: i.e. token:  {token}' '{calculate_glove_similarity(i,word_to_find)} '")

                            if (i == word_to_find or similarity > 0.9)  :
                        #print(f"At {index}, word: {i},checked with synonyms of {token}, i.e. {word_to_find}")
                        #print(f"distance: {calculate_glove_similarity(i,word_to_find.replace('_', ' ')) }")
                        #if i == word_to_find:# or  calculate_glove_similarity(i, word_to_find.replace("_", " ")) > 0.9:

                             suggestions.append((index,i,word_to_find,token))
                        #print(f"Similarity between {i} and {word_to_find}: {calculate_scibert_similarity(i,word_to_find)}")
                       #if :
                          #print(f"Similarity between {i} and {word_to_find}: {calculate_scibert_similarity(i,word_to_find)}")
                          #suggestions.append((index,word_to_find,token))

    unique_results = {}
    for item in suggestions:
        key = (item[0], item[1])
        if key not in unique_results:
            unique_results[key] = item


    # Convert the dictionary values back to a list
    suggestions = list(unique_results.values())
    return (suggestions)
