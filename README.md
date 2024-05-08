# MongoDB-LLM-Chatbot
Built a chatbot that retrieves data from a MongoDB using the LangChain LLM model, enhancing data accessibility and user interaction.


In the Python scripts you provided, you are building a retrieval-based question answering system that integrates MongoDB, OpenAI's language models, and Gradio for a web interface. This setup is designed to load documents from a directory into MongoDB, process them into embeddings using OpenAI's models, and then utilize these embeddings to perform a similarity search to retrieve and answer queries. Here's a breakdown of the processes within each script:
1) load.py
Data Loading and Processing: The script loads text files from a specified directory into a MongoDB collection using DirectoryLoader from langchain.document_loaders.
Embedding Generation: It converts the text data into vector embeddings using OpenAIEmbeddings which is helpful for efficient retrieval and similarity searches.
Vector Storage: The embedded vectors are stored in MongoDB using MongoDBAtlasVectorSearch, allowing for fast retrieval based on semantic similarity._

2) extraction.py
Retrieval Setup: Connects to MongoDB and initializes the vector store with pre-computed embeddings.
Query Processing Function: This function takes a user's query, finds the most relevant document using semantic similarity search, and then uses an OpenAI-based model for further processing or answering questions based on the retrieved content.
User Interface with Gradio: Provides a web-based interface where users can input questions, receive answers based on the data retrieved from MongoDB, and see how the model processes and answers using both direct retrieval and OpenAI's advanced NLP capabilities.
