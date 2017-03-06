# -*- coding: utf-8 -*-
import scrapy


class ExampleSpider(scrapy.Spider):
    name = 'lokago_spider'
    start_urls = ['http://www.golf.de/publish/turnierkalender/alle-turniere/kalender/info?tournamentid=490001001411&clubid=3316']

    def parse(self, response):
        SET_SELECTOR = '.grid_10'
        for div in response.css(SET_SELECTOR):

            NAME_SELECTOR = 'table'
            table = div.css(NAME_SELECTOR).extract_first()

            for tr in div.css('tr'):
                for td in div.css('td ::text'):
                    # tr = tr.select('tr').extract()
                    yield {'tr': {
                        'td1': td
                    }}


            # yield {
            #     'table': table.css(NAME_SELECTOR).extract_first()
            # }
