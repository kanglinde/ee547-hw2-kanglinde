Harry Kang

lindekan@usc.edu

Problem 1:
- When running ./test.sh, used "python3 -m json.tool" instead of "python -m json.tool". Otherwise, it would produce "python: command not found" with exit code 127.

Problem 2:
- "numpy" (included in requirements.txt) is installed only because "pytorch" gives a warning "Failed to initialize NumPy". The assignment itself(train_embeddings.py) does NOT need "numpy".

Problem 3:
- Had to use "python3" instead of "python" when running ./test.sh. Should change "python" command version accordingly while testing.