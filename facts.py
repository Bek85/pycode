from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

# Initialize the embeddings
embeddings = OpenAIEmbeddings()

# emb = embeddings.embed_query("Happy!")

# print(emb)

# Initialize the text splitter
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

# Load and split the documents
loader = TextLoader("facts.txt")
documents = loader.load_and_split(text_splitter)

# for doc in documents:
#     print(doc.page_content)
#     print("\n")


# Embed and store the documents
db = Chroma.from_documents(documents, embedding=embeddings, persist_directory="emb")

# Search for similar documents
results = db.similarity_search(
    "What is an interesting fact about the English language?"
)

for result in results:
    print("\n")
    print(result.page_content)
