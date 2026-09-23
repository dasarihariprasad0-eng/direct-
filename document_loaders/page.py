from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
url = "https://routes2roots.com/what-is-a-digital-learning-art-program-benefits-for-students-in-india/?utm_source=Digitales&utm_medium=GoogleAds&utm_campaign=Traffic&gad_source=1&gad_campaignid=23938532046&gbraid=0AAAAAC_mGyNADMZUdBWy8mnr8n4cogQ0u&gclid=Cj0KCQjwzY7VBhDwARIsAFtPvBQTWLh911SMkYni38M4OqIp16H4h0hxXOJj2Ey6TJI4DyYgVhqGjRsaAvEAEALw_wcB"
loader = WebBaseLoader(url)
docs = loader.load()
print(docs[0])