# wanger.dev

基于 [Hugo](https://gohugo.io/) + [Blowfish](https://blowfish.page/) 主题搭建的个人博客，部署在 GitHub Pages，绑定自定义域名 [wanger.dev](https://wanger.dev/)。

内容覆盖三个方向：

- **LLM** — AI 搜索与大语言模型方向的系统性梳理
- **摄影** — 摄影基础知识、学习笔记
- **写作** — 阅读与写作积累

## 项目结构

```
.
├── config/
│   └── _default/
│       ├── hugo.toml              # Hugo 主配置（站点地址、分类、分页等）
│       ├── params.toml            # Blowfish 主题参数
│       ├── markup.toml            # Markdown 渲染配置（Goldmark、KaTeX 等）
│       ├── languages.zh-cn.toml   # 中文语言配置
│       └── menus.zh-cn.toml       # 导航菜单
├── content/
│   ├── about/                     # 关于页面
│   ├── llm/                       # LLM 方向文章
│   ├── photography/               # 摄影方向文章
│   └── writing/                   # 写作方向文章
├── assets/                        # 站点级资源（图片等）
├── layouts/
│   └── partials/                  # 自定义 partial 模板覆盖
├── static/                        # 静态文件（CNAME 等）
├── themes/
│   └── blowfish/                  # Blowfish 主题
└── public/                        # Hugo 构建输出（不提交）
```

每篇文章采用 [Page Bundle](https://gohugo.io/content-management/page-bundles/) 组织，即 `content/<section>/<slug>/index.md`，文章配图放在同目录下。

## 本地开发

### 环境要求

- [Hugo Extended](https://gohugo.io/installation/) >= 0.141.0（需要 extended 版本以支持 SCSS）
- Git

macOS 安装：

```bash
brew install hugo
```

### 启动本地服务

```bash
# 克隆仓库（含主题子模块）
git clone --recurse-submodules https://github.com/Whiplashzeb/Whiplashzeb.github.io.git
cd Whiplashzeb.github.io

# 启动开发服务器（包含草稿）
hugo server -D
```

浏览器访问 `http://localhost:1313` 即可预览，修改文件后自动热更新。

### 新建文章

```bash
# 在对应 section 下创建文章目录
mkdir -p content/llm/my-new-post
```

然后在目录中创建 `index.md`，frontmatter 示例：

```toml
+++
title = "文章标题"
description = "文章描述"
date = 2026-04-01
tags = ["标签1", "标签2"]
categories = ["LLM"]
showTableOfContents = true
+++
```

### 构建与部署

```bash
# 本地构建
hugo

# 输出在 public/ 目录
```

推送到 `master` 分支后，GitHub Actions 自动部署到 GitHub Pages。
