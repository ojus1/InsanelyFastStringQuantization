'''
Python implementation of 
    "Extremely Fast Text Feature Extraction for Classification and Indexing" 
    https://www.hpl.hp.com/techreports/2008/HPL-2008-91R1.pdf
'''
import random
from typing import List, Union
from copy import deepcopy

def extractWordSet(text, num_bits, codetable):
    text = text.lower().encode('utf-8')
    vec = [0] * num_bits
    wordhash = 0
    for ch in text:
        code = codetable[ch]
        if code != 0:
            wordhash = (wordhash>>1) + code
        else:
            if wordhash != 0:
                vec[wordhash % num_bits] = 1
                wordhash = 0
    return vec

def prepTable(table_size=int(2**20), random_table=True, seed=12345):
    if random_table:
        random.seed(seed)
        rand = [random.randint(0, 255) for _ in range(256)]
    else:
        rand = [319, 784, 755, 591, 680, 1014, 726, 323, 425, 254, 561, 910, 105, 166, 704, 348, 1011, 502, 557, 817, 917, 566, 980, 620, 224, 813, 340, 91, 131, 704, 522, 288, 589, 603, 504, 922, 12, 626, 314, 747, 929, 871, 271, 986, 828, 7, 454, 773, 269, 426, 1003, 571, 334, 916, 933, 81, 363, 70, 856, 813, 220, 57, 614, 773, 204, 197, 763, 248, 730, 7, 834, 501, 367, 1016, 312, 903, 153, 10, 797, 939, 445, 395, 734, 499, 675, 574, 659, 801, 703, 699, 521, 655, 787, 537, 502, 778, 519, 197, 317, 34, 1012, 310, 421, 710, 779, 161, 381, 847, 856, 857, 348, 356, 676, 43, 491, 632, 553, 38, 13, 429, 106, 446, 591, 16, 742, 71, 174, 190, 60, 881, 499, 12, 612, 951, 69, 427, 596, 343, 20, 651, 502, 621, 347, 380, 306, 634, 472, 140, 256, 161, 476, 782, 480, 907, 612, 36, 133, 48, 253, 982, 626, 885, 372, 640, 333, 923, 541, 163, 625, 555, 118, 501, 105, 804, 332, 462, 818, 173, 1012, 132, 459, 253, 134, 745, 104, 517, 399, 107, 66, 863, 343, 477, 232, 206, 566, 93, 923, 598, 52, 167, 759, 476, 33, 796, 36, 398, 481, 760, 52, 109, 597, 422, 911, 931, 964, 765, 167, 128, 799, 11, 679, 850, 93, 367, 727, 923, 99, 526, 126, 491, 569, 65, 600, 75, 708, 563, 629, 702, 551, 871, 185, 81, 87, 715, 510, 959, 1015, 1, 618, 976, 803, 1011, 51, 1004, 214, 909]
    codetable = [0] * table_size
    for ch in range(table_size):
        if chr(ch).isalpha():
            idx = 0
            for byte in chr(ch).lower().encode("utf-8"):
                idx += byte
            codetable[ch] = rand[idx % len(rand)]
        else:
            codetable[ch] = 0
    return codetable

class Hasher:
    def __init__(self, feature_vector_dim, table_size=int(2**20), random_table=True, seed=12345):
        self.codetable = prepTable(table_size=table_size, random_table=random_table, seed=seed)
        self.fv_dim = feature_vector_dim

    def vectorize(self, inp: Union[List[str], str], progress_bar=False):
        if progress_bar:
            from tqdm import tqdm
        if isinstance(inp, str):
            return extractWordSet(inp, self.fv_dim, self.codetable)
        if progress_bar:
            out = [extractWordSet(item, self.fv_dim, self.codetable) for item in tqdm(inp)]
        else:
            out = [extractWordSet(item, self.fv_dim, self.codetable) for item in inp]
        return out
