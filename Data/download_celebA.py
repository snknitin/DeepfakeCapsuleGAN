from bs4 import BeautifulSoup
import urllib
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import time


def download_from_dropbox(url):

    # Get the content of the url page
    resp = urllib.request.urlopen(url)
    # Parse it with beautifulsoup
    soup = BeautifulSoup(resp, "lxml", from_encoding=resp.info().getparam('charset'))

    list_zip = []

    # Look for the links in the soup
    for link in soup.find_all('a', href=True):
        try:
            # Exploring the source code of said page shows
            # that the links I'm interested in have these properties
            if link["class"] == ["file-link"]:
                list_zip.append(link["href"])
        except KeyError:
            pass

    # Strip the "?dl=0" at the end of each link
    list_zip = [f.split("?dl=0")[0] for f in list_zip]

    # Function we'll map to the url so that the calls
    # are in parallel
    def call_wget(file_name):
        subprocess.call('wget ' + file_name, shell=True)

    pool = ThreadPool(4)  # Sets the pool size to 4
    # Open the urls in their own threads
    # and return the results
    pool.map(call_wget, list_zip)
    # close the pool and wait for the work to finish
    pool.close()
    pool.join()


if __name__ == '__main__':

    links = ["https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAq9krDJxUMh1m0hbbxdnl4a/Img/img_celeba.7z?dl=0"]

    start = time.time()
    for url in links:
        download_from_dropbox(url)
    print(time.time() - start)