import urllib.request
import urllib.parse
from bs4 import BeautifulSoup

containerRecord = {} 

for i in range(0,1000):
    url = 'http://127.0.0.1'
    f = urllib.request.urlopen(url)
    soup = BeautifulSoup(f.read().decode('utf-8'),"lxml")
    print(soup.get_text())
    final_content = soup.get_text()[4:]
    print(final_content)
    if str(final_content) not in containerRecord:
    	containerRecord[str(final_content)] = 1
    else:
    	containerRecord[str(final_content)]= containerRecord[str(final_content)] + 1 

for x, y in containerRecord.items():
	print(x,y) 


print(len(containerRecord))

	



