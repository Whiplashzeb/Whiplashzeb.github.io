# wanger.dev

基于 [Hugo](https://gohugo.io/) 与 [Blowfish](https://blowfish.page/) 主题的个人博客，部署在 GitHub Pages，绑定自定义域名 [wanger.dev](https://wanger.dev/)。

内容包括 LLM 主线、LLM 论文里程碑、摄影主线、摄影经典与随笔。四个系列栏目提供章节全图和阅读进度，文章页提供目录、选集导航与评论。

## 项目结构

```text
.
├── config/_default/        # 站点、主题、语言与导航配置
├── content/
│   ├── about/              # 关于页面
│   ├── llm/                # LLM 主线
│   ├── llm-papers/         # LLM 论文里程碑
│   ├── photography/        # 摄影主线
│   ├── photo-classics/     # 摄影经典
│   └── writing/            # 随笔
├── data/outlines/          # 系列章节与文章索引
├── assets/                 # 站点资源
├── layouts/                # 本站模板覆写
├── static/                 # 静态文件
├── themes/blowfish/        # 主题子模块
└── public/                 # 构建输出，不提交
```

文章采用 Hugo Page Bundle 组织：`content/<section>/<slug>/index.md`，图片放在文章同目录。

## 本地运行

需要 Git 与 Hugo Extended。Hugo 版本建议与 [.github/workflows/deploy.yml](.github/workflows/deploy.yml) 一致，当前为 0.159.1。

```bash
git clone --recurse-submodules https://github.com/Whiplashzeb/Whiplashzeb.github.io.git
cd Whiplashzeb.github.io
hugo server -D
```

浏览器访问 `http://localhost:1313` 查看预览。`-D` 包含草稿；正式构建默认不包含草稿和未来文章。

```bash
hugo --gc --minify
```

静态文件输出到 `public/`。推送 `master` 后，GitHub Actions 构建并部署到 GitHub Pages。

## 技术与外观

- Goldmark 解析 Markdown，KaTeX 显示数学公式。
- Blowfish 提供文章、分类、目录及响应式布局；本站在 `layouts/` 中维护模板覆写。
- Giscus 提供评论，Vanta.js Fog 提供动态背景，Cloudflare Web Analytics 提供访问统计。
- 支持亮暗模式，中文内容使用 CJK 语言配置。

主题通过 Git 子模块管理，本站的页面与样式调整保存在仓库自身的配置和模板中。
