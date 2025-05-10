from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper, MAX_TOKENS

load_dotenv()

client = OpenAI()
tokenizer = OpenAITokenizerWrapper()

# Data extraction
converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")

# Apply hybrid chunking
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

print(len(chunks))
