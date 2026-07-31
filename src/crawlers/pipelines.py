import json
import os


class NovelSpiderPipeline:
    def open_spider(self, spider):
        """爬虫启动时执行"""
        self.all_chapters = []
        # 确保目录存在
        os.makedirs('../data/processed', exist_ok=True)

    def close_spider(self, spider):
        """爬虫关闭时执行"""
        # 保存元数据
        if self.all_chapters:
            metadata = {
                "total_chapters": len(self.all_chapters),
                "chapters": self.all_chapters
            }

            with open('../data/processed/novel_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"💾 元数据保存完成: {len(self.all_chapters)} 个章节")

    def process_item(self, item, spider):
        """处理每个章节项目"""
        # 这里可以添加额外的处理逻辑
        # 比如数据清洗、验证等
        chapter_data = {
            'title': item.get('title'),
            'page_count': item.get('page_count'),
            'url': item.get('url')
        }

        self.all_chapters.append(chapter_data)
        return item