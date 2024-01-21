import os

P = "test_shoulder/results/test/0"

files = os.listdir(P)

for file in files:
    fullP = os.path.join(P, file)
    print(file)
    file = file.replace("Out_", "")
    newfile = os.path.join(P, file)
    os.rename(fullP, newfile)