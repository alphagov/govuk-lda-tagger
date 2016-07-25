import csv

print("Reading input file 'input/audits_with_content.csv'")
with open('input/audits_with_content.csv', 'r') as f:
    reader = csv.reader(f)
    documents = list(reader)

titles = list()

print("Generating titles for all documents")
for index, document in enumerate(documents):
    link = document[0]
    slug = link.split('/')[-1]
    title = slug.replace('-', ' ')
    titles.append({'id': index, 'title': title})

print("Writing titles into output file")
with open('output/data.titles', 'w') as csvfile:
    fieldnames = ['id', 'title']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for title_record in titles:
        writer.writerow(title_record)
