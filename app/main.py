from app.pdf.pdf_processor import PDFExtraction
from app.text_preprocessing.text_chunking import TextChunker
from app.model.tokenize import CharTokenizer
from app.model.lstm_model import LSTMModel
from app.model.qa_dataset import QADataset
from app.model.trainer import Trainer
from app.chatbot.qa_service import QAService
from pathlib import Path


PDF_PATH = r"C:\Users\Priya Bhaskar\OneDrive\Documents\project_6_chat_lstm\chatbot_lstm\app\pdf\-0___jonew__judis__10796.pdf"
WEIGHTS_PATH = "artifacts/lstm_model.weights.h5"
TOKENIZER_PATH = "artifacts/tokenizer.json"


def train():

    # step 1: Extract the text page by page
    extractor = PDFExtraction()
    pages = extractor.extract_text(PDF_PATH)

    # step 2: Chunk each page separately
    chunker = TextChunker(max_chars=750, overlap=100)
    chunks = []
    for page_text in pages:
        chunks.extend(chunker.chunk(page_text))

    
    # step 3: create synthetic data of QA pairs
    qa_pairs = chunker.make_qas(chunks=chunks)

    # step 4: tokenize and build the dataset
    tokenizer = CharTokenizer()
    dataset = QADataset(tokenizer=tokenizer)
    X, Y = dataset.build_arrays(qa_pairs=qa_pairs)

    #step 5: Build the LSTM and train the model
    model = LSTMModel(vocab_size=len(tokenizer.vocab))
    trainer = Trainer(model=model, tokenizer=tokenizer)
    trainer.fit(X=X, Y=Y, epochs=5, batch_size=16)
    trainer.save_all(weights_path=WEIGHTS_PATH, tokenizer_path=TOKENIZER_PATH)




def chat():
    # load the tokenizer and model
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    model = LSTMModel(vocab_size=len(tokenizer.vocab))
    model.load_weights(WEIGHTS_PATH)
    qa = QAService(tokenizer=tokenizer, model=model)

    print("\nJudgement Bot is ready. Type your questions or exit to quit. \n")

    while True:
        q = input("Ask a question: ")
        if q.lower().strip == "exit":
            print("GoodBye!")
            break

        answer = qa.answer(q)
        print("Answer:", answer)

    
if __name__ == "__main__":
    if not Path(WEIGHTS_PATH).exists():
        print("Training the LSTM model....")

        train()
    
    chat()


    







