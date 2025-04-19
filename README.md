# Read Along - A Reader's Companion

## Overview
**Read Along** is a chatbot application designed to enhance the reading experience by providing additional context and answering questions based on the user’s progress in the book. It allows users to interact with the content in a more engaging, intelligent, and personalized way.

## Features
- **Contextual Information**: Read Along retrieves relevant background or supporting information based on the section of the book the user is currently reading.
- **Question Answering**: Users can ask questions about the book, and the system will respond using only the content up to the user’s current progress.
- **User-Friendly Interface**: The application is built with simplicity and clarity in mind, allowing users to focus on reading and interacting without distractions.
- **Dynamic Book Progression**: Tracks how much of the book has been read and limits the LLM's knowledge accordingly, preserving the reading experience.
- **EPUB Support**: Parses and understands `.epub` format books, including spine navigation and chapter mapping.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (FastAPI)
- **Large Language Model (LLM)**: `hermes-3b` or `hermes-3:8b`
- **Embeddings**: `intfloat/multilingual-e5-large`
- **Vector Database**: Pinecone
- **Deployment**: REST API

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Harish892002/daraga.git
   cd daraga
   ```

2. **Set Up the Environment**
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Create a `.env` file with:
     ```env
     PINECONE_API_KEY=your_key_here
     ```

3. **Run the Application**
   ```bash
   uvicorn src.api:app --reload 
   ```

## Project Structure
```
src/
├── api.py                   # REST API logic
├── main.py                  # Main entrypoint
├── build_rag.py             # RAG pipeline setup
├── load_book.py             # Chapter and spine content loader
├── preprocess.py            # Sentence splitting, metadata tracking
└── loaders/
    └── epub_loader.py       # EPUB parsing logic
```
