from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

converter = DocumentConverter()

# Basic PDF extraction
paper_url = "https://arxiv.org/pdf/2408.09869"
result = converter.convert(paper_url)

document = result.document
markdown = document.export_to_markdown()
json = document.export_to_dict()
print(markdown)

# Basic HTML extraction
page_url = "https://google.github.io/adk-docs/"
result = converter.convert(page_url)

document = result.document
markdown = document.export_to_markdown()
print(markdown)


# Scrape multiple pages using sitemap
sitemap_urls = get_sitemap_urls(page_url)
conv_result_iter = converter.convert_all(sitemap_urls)

docs = []
for conv_result in conv_result_iter:
    if conv_result.document:
        document = conv_result.document
        docs.append(document)

print(docs)
