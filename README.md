Harry Kang

lindekan@usc.edu

Problem 1:
- When running ./test.sh, use "python3 -m json.tool" instead of "python -m json.tool". Otherwise, it produces "python: command not found" with exit code 127.

Problem 2:
- "numpy" (included in requirements.txt) is installed only because "pytorch" gives a warning "Failed to initialize NumPy". The assignment itself(train_embeddings.py) does NOT need "numpy".