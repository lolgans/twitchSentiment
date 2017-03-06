# -*- coding: utf-8 -*-
import scrapy


class ExampleSpider(scrapy.Spider):
    name = "example"
    allowed_domains = ["example.com"]
    start_urls = ['http://example.com/']

    def parse(self, response):
        pass





# class BrickSetSpider(scrapy.Spider):
#     name = 'lokago_spider'
#     start_urls = ['http://www.golf.de/publish/turnierkalender/alle-turniere/kalender/info?tournamentid=490001001411&clubid=3316']
#
#     def parse(self, response):
#         SET_SELECTOR = '.grid_10'
#         for table in response.css(SET_SELECTOR):
#
#             NAME_SELECTOR = 'table'
#             # PIECES_SELECTOR = './/dl[dt/text() = "Pieces"]/dd/a/text()'
#             # MINIFIGS_SELECTOR = './/dl[dt/text() = "Minifigs"]/dd[2]/a/text()'
#             # IMAGE_SELECTOR = 'img ::attr(src)'
#             yield {
#                 'table': table.css(NAME_SELECTOR).extract_first()
#             }
#
#         # NEXT_PAGE_SELECTOR = '.next a ::attr(href)'
#         # next_page = response.css(NEXT_PAGE_SELECTOR).extract_first()
#         # if next_page:
#         #     yield scrapy.Request(
#         #         response.urljoin(next_page),
#         #         callback=self.parse
#         #     )