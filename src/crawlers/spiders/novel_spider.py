import os

import scrapy
import re

class NovelSpider(scrapy.Spider):
    name = 'novel'
    start_urls = ['https://www.aikansu8.com/book/165623/index.html']

    def __init__(self, *args, **kwargs):
        super(NovelSpider, self).__init__(*args, **kwargs)
        self.chapter_titles = {}  # 存储章节号和标题的映射


    def parse(self, response):
        """解析目录页，获取所有章节链接"""
        chapters = response.css('#list dd a')
        print(f"🎯 找到 {len(chapters)} 个章节")

        # 首先收集所有章节的标题
        for i, chapter in enumerate(chapters):
            title = chapter.css('::attr(title)').get()
            if title:
                chapter_num = self.get_chapter_number(title)
                if chapter_num > 0:
                    self.chapter_titles[chapter_num] = title
                else:
                    self.chapter_titles[i + 1] = title  # 如果没有章节号，使用序号


        for chapter in chapters[:1]:
            title = chapter.css('::attr(title)').get()
            href = chapter.css('::attr(href)').get()
            print(title)
            if title and href:
                full_url = response.urljoin(href)
                print(f"📖 开始抓取: {title}")
                yield scrapy.Request(
                    url=full_url,
                    callback=self.parse_chapter,
                    meta={
                        'chapter_title': title,
                        'chapter_content': [],
                        'current_page': 1,
                        'is_first_page': True  # 标记是第一页
                    }
                )

    def parse_chapter(self, response):
        """解析章节内容页"""
        chapter_title = response.meta['chapter_title']
        is_first_page = response.meta.get('is_first_page', False)

        # 提取内容
        content_lines = response.css('#content p::text').getall()
        print(content_lines)
        cleaned_content = []
        for line in content_lines:
            text = line.strip()
            cleaned_content.append(text)

        current_content = response.meta['chapter_content'] + cleaned_content

        # 处理文件保存
        if is_first_page:
            # 第一页，创建新文件
            filepath = self.create_chapter_file(chapter_title)
        else:
            # 后续页面，获取文件路径
            filepath = self.get_chapter_filepath(chapter_title)

        # 保存内容到文件
        self.save_to_file(filepath, cleaned_content, is_first_page)

        # 检查下一页
        next_link = response.css('#pager_next::attr(href)').get()

        if next_link:
            full_next_url = response.urljoin(next_link)

            # 判断是同一章还是新章节
            current_id = self.get_chapter_id(response.url)
            next_id = self.get_chapter_id(full_next_url)

            if current_id == next_id:
                print("➡️ 同一章节的下一页")
                yield scrapy.Request(
                    url=full_next_url,
                    callback=self.parse_chapter,
                    meta={
                        'chapter_title': chapter_title,
                        'chapter_content': current_content,
                        'current_page': response.meta['current_page'] + 1,
                        'is_first_page': False  # 不是第一页
                    }
                )
            else:
                print("🎯 >>>>> 发现下一章！ <<<<<")
                # 保存当前章节
                yield {
                    'title': response.meta['chapter_title'],
                    'content': '\n'.join(current_content),
                    'page_count': response.meta['current_page'],
                    'url': response.url
                }

                # 获取下一章的完整标题
                next_chapter_title = self.get_next_chapter_full_title(chapter_title)
                # 跳转到下一章
                yield scrapy.Request(
                    url=full_next_url,
                    callback=self.parse_chapter,
                    meta={
                        'chapter_title': next_chapter_title,
                        'chapter_content': current_content,
                        'current_page': 1,
                        'is_first_page': True  # 是新章节的第一页
                    }
                )
        else:
            print("✅ 章节完成")

            yield {
                'title': response.meta['chapter_title'],
                'content': '\n'.join(current_content),
                'page_count': response.meta['current_page'],
                'url': response.url
            }

    def get_chapter_id(self, url):
        """从URL提取章节ID"""
        match = re.search(r'/book/\d+/(\d+)', url)
        return match.group(1) if match else None

    def get_chapter_number(self, title):
        """从标题提取章节号"""
        match = re.search(r'第(\d+)章', title)
        return int(match.group(1)) if match else 0

    def get_next_chapter_title(self, current_title):
        """生成下一章标题（只包含章节号）"""
        current_num = self.get_chapter_number(current_title)
        if current_num > 0:
            return f"第{current_num + 1}章"
        return current_title

    def get_next_chapter_full_title(self, current_title):
        """获取下一章的完整标题"""
        current_num = self.get_chapter_number(current_title)
        if current_num > 0:
            next_num = current_num + 1
            # 从预先收集的标题中查找
            if next_num in self.chapter_titles:
                return self.chapter_titles[next_num]
            else:
                return f"第{next_num}章"
        return current_title

    def create_chapter_file(self, chapter_title):
        """创建新章节文件并返回文件路径"""
        # 确保目录存在
        os.makedirs('../data/raw/', exist_ok=True)

        chapter_num = self.get_chapter_number(chapter_title)
        if chapter_num > 0:
            # 使用完整的章节标题作为文件名
            safe_title = re.sub(r'[\\/*?:"<>|]', '', chapter_title)
            filename = f"第{chapter_num:02d}章 {safe_title}.txt"
        else:
            # 如果没有章节号，使用标题
            safe_title = re.sub(r'[\\/*?:"<>|]', '', chapter_title)[:20]
            filename = f"{safe_title}.txt"

        filepath = f"../data/raw/{filename}"

        print(f"📁 创建新章节文件: {filename}")

        # 写入章节标题
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{chapter_title}\n")

        return filepath

    def get_chapter_filepath(self, chapter_title):
        """根据章节标题生成文件路径"""
        chapter_num = self.get_chapter_number(chapter_title)
        if chapter_num > 0:
            # 使用完整的章节标题作为文件名
            safe_title = re.sub(r'[\\/*?:"<>|]', '', chapter_title)
            filename = f"第{chapter_num:02d}章 {safe_title}.txt"
        else:
            # 如果没有章节号，使用标题
            safe_title = re.sub(r'[\\/*?:"<>|]', '', chapter_title)[:20]
            filename = f"{safe_title}.txt"

        return f"../data/raw/{filename}"

    def save_to_file(self, filepath, content_lines, is_first_page=False, chapter_title=""):
        """保存内容到文件"""
        try:
            mode = 'a'  # 总是使用追加模式，因为标题已经在create_chapter_file中写入了

            with open(filepath, mode, encoding='utf-8') as f:
                for line in content_lines:
                    # 在保存时过滤，确保没有漏网之鱼
                    if not any(pattern in line for pattern in [
                        '本小章还未完', '请点击下一页', '继续阅读', '未完待续'
                    ]):
                        f.write(line + "\n")

            action = "创建并写入" if is_first_page else "追加到"
            print(f"💾 已{action}文件: {os.path.basename(filepath)}")

        except Exception as e:
            print(f"❌ 文件操作失败: {e}")