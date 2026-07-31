BOT_NAME = 'novelAgent'

SPIDER_MODULES = ['src.crawlers.spiders']
NEWSPIDER_MODULE = 'src.crawlers.spiders'

ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 5
# 爬虫设置
CONCURRENT_REQUESTS = 1
# 启用管道
ITEM_PIPELINES = {
    'crawlers.pipelines.NovelSpiderPipeline': 800,
}



USER_AGENT = 'novelAgent (用于AI智能体+知识管理)'
LOG_LEVEL = 'INFO'