-----------------------
Message Decoder
-----------------------
To run the program please use:
    python3 program.py d | b | i | u | a | g


The function to decode a secret message takes six inputs:
- A character (d, b, i, u, a, g) specifying the algorithm (DFS, BFS, IDS, UCS, A* and Greedy respectively)
- The name of a text file containing a secret message
- The name of a text file containing a list of words, in alphabetical order and each on a separate line, which will act as a dictionary of correct words
- A threshold, t, specifying what percentage of words must be correct for this to count as a goal (given as an integer between 0 and 100).
- A string containing the letters that are allowed to be swapped
- A character (y or n) indicating whether to print the messages corresponding to the first 10 expanded nodes.
