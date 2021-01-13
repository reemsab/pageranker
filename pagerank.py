import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

#
def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus)
    p = (1 - damping_factor)/n
    if(len(corpus[page]) ==0):
        p = 1/n
    numlinks = len(corpus[page])
    prob = dict()
    for pages in corpus:
        if pages in corpus[page]:
            prob[pages] = p + (damping_factor/ numlinks)
        else:
            prob[pages] = p
    return prob

#
def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pop = []
    prob = dict()
    for i in corpus:
        pop.append(i)
        prob[i] = 0
    p = random.sample(pop, k = 1) 
    currm = transition_model(corpus, p[0], damping_factor) 
    for i in range(n - 1) : 
        for i in pop:
            tm = transition_model(corpus, i, damping_factor)
            for j in pop:
                prob[j] = (prob[j] + (currm[i] * tm[j]))          
        for k in pop:
            currm[k]= prob[k]
            prob[k] = 0       
    return currm 


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    prob= dict()
    dif = 1
    for i in corpus:
        prob[i] = 1/len(corpus)
    while(dif >= 0.001):
        pro = dict()
        for p in corpus:
            sum = 0
            for i in corpus:
                if p in corpus[i]:
                    numlinks = n if len(corpus[i]) == 0 else  len(corpus[i])
                    sum = sum + (prob[i]/numlinks)
            pro[p] = ((1 - damping_factor)/n) + ( damping_factor * sum)
            dif = min(abs(prob[p] - pro[p]), dif)
        prob = pro   
        
    return prob     
    raise NotImplementedError


if __name__ == "__main__":
    main()
