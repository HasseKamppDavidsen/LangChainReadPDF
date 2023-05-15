# %%
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import streamlit as st

# %%
# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview


# %%
# location of the pdf file/files.
# reader = PdfReader('Neutracon_data_sheet.pdf')

st.set_page_config(layout="wide")

st.title("Ask questions to your PDF using ChatGPT API")

with st.sidebar:
    st.image('https://mapbackgroundlayers.blob.core.windows.net/niraslogo/logo.png', width=150)
    keyStr = st.text_input("Write your API key for OpenAI")
    os.environ["OPENAI_API_KEY"] = keyStr
    st.markdown("----")
    model = st.radio("Choose model", ["text-davinci", "GPT-4-0314"])
    st.markdown("----")
    uploaded_file = st.file_uploader("Choose pdf to be used")
    
if len(keyStr) > 0:

    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        # read data from the file and put them into a variable called raw_text
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # We need to split the text that we read into smaller chunks so that durings
        # information retreival we don't hit the token size limits.
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)

    # Check model type
    if model == "text-davinci":
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
    else:
        chat = ChatOpenAI(model_name="gpt-4", temperature=0)
        chain = load_qa_chain(chat, chain_type="stuff")


    # %%
    # Referat
    query = st.text_area("Add promt:")

    if len(query) > 0 and uploaded_file is not None:
        docs = docsearch.similarity_search(query)

        answer = chain.run(
                            input_documents=docs,
                            question=query)

        st.header("Markdown")
        st.markdown(answer, unsafe_allow_html=True)
        # st.header("Code")
        # st.code(answer)

    st.header("Product Data sheet")
    st.code("""
    Find out if there is a health risk connected to the product.
    """)

    st.code("""
    Find out how many chemicals the product is made of and find the CAS number of the chemicals that make op the composition of the product.

    Give me the answer structured as a html table with Chemical name and CAS number as titles
    """)

    st.header("Municipal meeting minutes")
    st.code("""
    Find alle de steder byen Stenløse nævnes i teksten. Strukturer svaret som en HTML tabel med en opsummering af hvad
    afsnittet af teksten handler om. Lav resumeet på 20 ord
    """)
    st.code("""
    Find investeringer vedtaget på over 1 million kroner.
    Strukturer svaret som en HTML tabel med navn på
    - investering
    - investeret beløb
    - personer der har vedtaget investeringen
    - beskrivelse af investeringen på 20 ord
    """)
