import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.evaluation.qa import QAGenerateChain


_ = load_dotenv(find_dotenv()) # read local .env file

llm_model = "gpt-3.5-turbo"
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))

new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

examples += new_examples

qa.run(examples[0]["query"])