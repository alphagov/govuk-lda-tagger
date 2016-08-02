import scrapy
import data
import html2text
import ipdb

class GovukSpider(scrapy.Spider):
    name = 'govuk'
    start_urls = data.dev_urls()

    def parse(self, response):
        xpath_selector = scrapy.Selector(response)
        div = xpath_selector.xpath('//main[@id="content"]').extract()[0]
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        text = converter.handle(div).replace('\n', ' ')

        yield {
            'url': response.url,
            'text': text
        }
