from InsanelyFastStringQuantization import Hasher
import random
import time

obj = Hasher(16, random_table=False)

def make_strings():
    with open("lorem_ipsum.txt", 'r') as f:
        corpus = " ".join([item.strip() for item in filter(lambda x: True if x != "\n" else False, f.readlines())])
        corpus = corpus.split(" ")
    lines = []
    for i in range(len(corpus)):
        try:
            num_words = random.randint(3, 10)
            lines.append(" ".join(corpus[i: i+num_words]))
        except:
            break
    # print(lines[:5], len(lines))
    return lines

if __name__ == "__main__":
    lines = make_strings()
    lines = lines * 300

    tick = time.time()
    obj.vectorize(lines, progress_bar=True)
    tock = time.time()
    print(f"Number of strings: {len(lines)}")
    l = 0
    for item in lines:
        l += len(item) / len(lines)
    print(f"Average String Length: {l}")
    print(f"Characters per Second: {(len(lines) * l) / (tock-tick)}")
    print("Time taken for Single Core:", tock-tick)
