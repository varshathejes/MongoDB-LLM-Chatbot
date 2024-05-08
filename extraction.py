from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_params

# Connect to MongoDB
client = MongoClient(key_params.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_data"
collection = client[dbName][collectionName]

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=key_params.openai_api_key)

# Create MongoDBAtlasVectorSearch with the MongoDB collection and embeddings
vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

# Define the function to query data
def query_data(query):

    # Perform similarity search using MongoDBAtlasVectorSearch
    docs = vectorStore.similarity_search(query, K=1)

    as_output = docs[0].page_content
    

    # Initialize OpenAI language model
    llm = OpenAI(openai_api_key=key_params.openai_api_key, temperature=0)

    # Convert the vector store to a retriever
    retriever = vectorStore.as_retriever()

    # Create a RetrievalQA model
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    # Perform question answering using the RetrievalQA model
    retriever_output = qa.run(query)


    # Return the outputs
    return as_output, retriever_output

# Create a Gradio interface
with gr.Blocks(theme=Base(), title="Chatbot") as demo:
    # Display a markdown message
    gr.Markdown(
        """
        Question answering
        """)

    # Input textbox for the user's question
    textbox = gr.Textbox(label="Enter your question:")

    # Submit button
    with gr.Row():
        button = gr.Button("Submit", variant="primary")

    # Output textboxes
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Output with Atlas")
        output2 = gr.Textbox(lines=1, max_lines=10, label="Output with Atlas and RAG and Chatgpt")

    # Specify the function to be called on button click and the associated inputs and outputs
    button.click(query_data, textbox, outputs=[output1, output2])

# Launch the Gradio interface
demo.launch()
