import lzma

file = "D:/LLM/openwebtext/openwebtext/urlsf_subset01-1_data.xz"
data = lzma.open(file, "rb").read()

print(data[:200])