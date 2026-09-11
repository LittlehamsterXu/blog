# Little Hamster 的博客

一个用 Hugo 和 PaperMod 搭建的小博客，记录技术实践、学习过程和日常想法。

线上地址：<https://blog.little-hamster-xu.workers.dev/>

## 本地运行

```powershell
hugo server --buildDrafts
```

打开 <http://localhost:1313/> 预览。

如果 Windows 下遇到缓存路径问题，可以使用项目内缓存：

```powershell
hugo --cacheDir .hugo_cache --gc --minify
```

## 网站结构

- `/posts/`：文章
- `/archives/`：归档
- `/search/`：搜索
- `/about/`：关于

其他栏目暂不放在公开站点，等内容准备好后再逐步补充。

## 新建文章

```powershell
hugo new content posts/my-post.md
```

文章写在 `content/posts/` 中。需要数学公式的文章，在 front matter 中加入：

```yaml
math: true
```

站点支持代码高亮、代码复制、KaTeX 数学公式、Mermaid 图表和 Callout 提示框。

## Cloudflare Workers 部署

这个仓库使用 Workers Builds 部署 Hugo 静态站点：

```text
Root directory: /
Build command: hugo --gc --minify
Deploy command: npx wrangler deploy
HUGO_VERSION: 0.166.0
```

`wrangler.jsonc` 已将 Workers Static Assets 目录固定为 `./public/`，不需要设置 Pages 的 Output directory。在 Cloudflare 项目的 `Settings > Build` 中确认 Deploy command 为 `npx wrangler deploy`，并在 `Settings > Environment variables` 中为 Production 和 Preview 添加 `HUGO_VERSION=0.166.0`。

推送到 `main` 后会自动构建和发布。正式地址配置在 `hugo.yaml` 中。

## 维护

- `themes/PaperMod` 保存为当前公开快照，不包含主题仓库历史。
- 站点扩展优先放在 `layouts/`、`assets/`、`data/` 和 `content/` 中。
- 新栏目先积累内容，再加入导航。
