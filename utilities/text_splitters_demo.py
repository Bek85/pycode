from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Sample text with various separators
sample_text = """
Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991.

Python emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.

Key features include:
- Dynamic typing
- Automatic memory management
- Large standard library
- Cross-platform compatibility

Python is widely used in web development, data analysis, artificial intelligence, and scientific computing.
"""

print("ORIGINAL TEXT:")
print("=" * 50)
print(sample_text)
print("\n")

# CharacterTextSplitter - only uses newline
char_splitter = CharacterTextSplitter(separator="\n", chunk_size=150, chunk_overlap=20)

char_chunks = char_splitter.split_text(sample_text)

print("CHARACTER TEXT SPLITTER RESULTS:")
print("=" * 50)
for i, chunk in enumerate(char_chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(repr(chunk))  # repr shows whitespace characters
    print(f"Readable: {chunk.strip()}")
    print("-" * 30)

print("\n")

# RecursiveCharacterTextSplitter - tries multiple separators
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

recursive_chunks = recursive_splitter.split_text(sample_text)

print("RECURSIVE CHARACTER TEXT SPLITTER RESULTS:")
print("=" * 50)
for i, chunk in enumerate(recursive_chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(repr(chunk))  # repr shows whitespace characters
    print(f"Readable: {chunk.strip()}")
    print("-" * 30)

print("\n")
print("COMPARISON SUMMARY:")
print("=" * 50)
print(f"CharacterTextSplitter created {len(char_chunks)} chunks")
print(f"RecursiveCharacterTextSplitter created {len(recursive_chunks)} chunks")

# Show how each handles the same content differently
print("\nKey differences:")
print("- CharacterTextSplitter: May break mid-sentence if newlines don't align well")
print(
    "- RecursiveCharacterTextSplitter: Tries to preserve sentence/paragraph boundaries"
)

# Demonstrate the separator priority
print("\n" + "=" * 50)
print("RECURSIVE SPLITTER SEPARATOR PRIORITY:")
print("=" * 50)
print("1. \\n\\n  (paragraph breaks)")
print("2. \\n    (line breaks)")
print("3. .     (sentence endings)")
print("4. !     (exclamations)")
print("5. ?     (questions)")
print("6. ,     (commas)")
print("7. ' '   (spaces)")
print("8. ''    (character by character - last resort)")
