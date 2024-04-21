from collections import defaultdict

import nltk
nltk.download('punkt')
nltk.download("stopwords")
import streamlit as st
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer
import io

def generateExtractiveSummary(text, n):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Remove stop words and punctuation
    stop_words = stopwords.words('english')
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Calculate the frequency of each word
    frequency = defaultdict(int)
    for word in words:
        frequency[word] += 1

    # Rank the sentences by their total word frequency
    rankings = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                rankings[i] += frequency[word]

    # Sort the sentences by their rankings and return the top n
    top_sentences = sorted(rankings, key=rankings.get, reverse=True)[:n]
    summary = [sentences[i] for i in top_sentences]
    return ' '.join(summary)

def generateAbstractiveSummary(inputText):
    # Load the fine-tuned model
    fine_tuned_model = BartForConditionalGeneration.from_pretrained("fine_tuned_bart_model")

    # Load the saved fine-tuned tokenizer
    tokenizer = BartTokenizer.from_pretrained("fine_tuned_bart_tokenizer")
    # Tokenize and encode the text
    inputs = tokenizer(inputText, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = fine_tuned_model.generate(inputs.input_ids, 
                                            max_length=150,  
                                            min_length = 40, 
                                            num_beams=4, 
                                            length_penalty=3.0, 
                                            early_stopping=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary;
    
def main():
    st.title("Summarizer")
    
    # Sidebar options
    page = st.sidebar.radio(
        "Select a page:", ("Summarize Random Text", "Summarize Document"));
    
    if(page=="Summarize Random Text"):
        generate_summary_from_random_text()
    else:
        summarize_document_page()
    

def generate_summary_from_random_text():
    # Radio buttons for choosing summarization type
    summarization_type = st.radio("Choose Summarization Type", ("Extractive", "Abstractive"))
    if(summarization_type == "Extractive"):
        st.header("Generate Extractive Summarization")
    else:
        st.header("Generate Abstractive Summarization")
        
    # Text input
    inputText = st.text_area("", height=300, placeholder="Enter some text to summarize...")
    if(summarization_type == "Extractive"):
        # Summarize random text
        if st.button("Summarize"):
            if inputText:
                generated_extractive_summary = generateExtractiveSummary(inputText, 2);
                st.success("Summary:\n" + generated_extractive_summary)
            else:
                st.warning("Please enter some text to summarize.")
                
    else:
        # Button to trigger summary generation
        if st.button("Summarize"):
            if inputText.strip() != "":
                summary = generateAbstractiveSummary(inputText)
                # Display the summary
                st.success("Summary:\n" + summary)
            else:
                st.warning("Please enter some text to summarize.")
    
def summarize_document_page():
    st.header("Summarize Document")
    
    # Radio buttons for choosing summarization type
    summarization_type = st.radio("Choose Summarization Type", ("Extractive", "Abstractive"))

    # File input
    file = st.file_uploader("Upload a file", type=["txt", "html", "pdf"])

    # Summarize document contents
    if st.button("Summarize") and file is not None:
        if file.type == "text/plain":
            # Read text file
            text = io.TextIOWrapper(file, encoding='utf-8').read()
            summary = generateExtractiveSummary(text, 3) if summarization_type == 'Extractive' else generateAbstractiveSummary(text)
            st.success("Summary:\n" + summary)
        elif file.type == "text/html":
            # Read HTML file
            html = io.TextIOWrapper(file, encoding='utf-8').read()
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            summary = generateExtractiveSummary(text, 3) if summarization_type == 'Extractive' else generateAbstractiveSummary(text)
            st.success("Summary:\n" + summary)
        elif file.type == "application/pdf":
            # Read PDF file
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            summary = generateExtractiveSummary(text, 3) if summarization_type == 'Extractive' else generateAbstractiveSummary(text)
            st.success("Summary:\n " + summary)
        else:
            st.warning("Invalid file format. Please upload a valid text, HTML, or PDF file.")

if __name__ == "__main__":
    main()
