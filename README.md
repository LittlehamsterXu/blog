# LittleHamster Xu 的博客

一个用 Hugo 和 PaperMod 搭建的小博客，记录技术实践、学习过程和日常想法。

线上地址：<https://blog.little-hamster-xu.workers.dev/>

## 本地运行

### Linux / WSL

使用 Hugo Extended **0.166.0**，与部署配置一致。PaperMod 主题已随仓库提供，无需安装 Node.js 或 npm 依赖。

首次安装（适用于 Linux x86_64，需要 `curl`、`tar` 和 `sha256sum`）：

```bash
mkdir -p .local/bin
(
  set -eu
  download_dir="$(mktemp -d)"
  trap 'rm -rf "$download_dir"' EXIT
  release_url="https://github.com/gohugoio/hugo/releases/download/v0.166.0"
  archive="hugo_extended_0.166.0_linux-amd64.tar.gz"
  curl -fL --retry 3 "$release_url/$archive" -o "$download_dir/$archive"
  curl -fL --retry 3 "$release_url/hugo_0.166.0_checksums.txt" -o "$download_dir/checksums.txt"
  (cd "$download_dir" && grep " $archive\$" checksums.txt | sha256sum -c -)
  tar -xzf "$download_dir/$archive" -C .local/bin hugo
)
```

在项目目录启动预览：

```bash
bash scripts/dev.sh
```

打开 <http://localhost:1313/>，修改文章后自动刷新；按 `Ctrl+C` 停止。

如果 1313 已被占用，可运行 `bash scripts/dev.sh --port 1314`，然后打开 <http://localhost:1314/>。

构建正式站点或新建文章：

```bash
bash scripts/hugo.sh --gc --minify
bash scripts/hugo.sh new content posts/my-post.md
```

本地程序保存在 `.local/bin/`，缓存保存在 `.hugo_cache/`，均已忽略提交。

### Windows / 已全局安装 Hugo

```powershell
hugo server --buildDrafts --baseURL http://localhost:1313/
```

打开 <http://localhost:1313/> 预览。

本地预览显式使用 `localhost` 作为 `baseURL`，这样文章封面等通过绝对地址生成的资源也会从本地加载；线上构建仍使用 `hugo.yaml` 中的正式地址。

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

文章写在 `content/posts/` 中。公式样式会根据正文自动加载，也可以在 front matter 中显式启用：

```yaml
math: true
```

站点支持代码高亮、代码复制、KaTeX 数学公式、Mermaid 图表和 Callout 提示框。

## Cloudflare Workers 部署

这个仓库使用 Workers Builds 部署 Hugo 静态站点：

```text
Root directory: /
Build command: bash scripts/build-cloudflare.sh
Deploy command: npx wrangler deploy
HUGO_VERSION: 0.166.0
```

`wrangler.jsonc` 已将 Workers Static Assets 目录固定为 `./public/`，不需要设置 Pages 的 Output directory。在 Cloudflare 项目的 `Settings > Build` 中确认 Deploy command 为 `npx wrangler deploy`，并在 `Settings > Environment variables` 中为 Production 和 Preview 添加 `HUGO_VERSION=0.166.0`。

推送到 `main` 后会自动构建和发布。正式地址配置在 `hugo.yaml` 中。

将 Cloudflare 控制台的 Build command 更新为上面的命令后，构建会先检查站内链接、资源、RSS 和站点地图，失败时停止发布。需要构建环境提供 Python 3；检查脚本不依赖第三方 Python 包。仓库修改不会自动改变控制台已保存的命令。

## GitHub Pages 双站发布

保留 `.github/workflows/deploy-github-pages.yml` 与 Cloudflare Workers Builds 并行发布。GitHub 流程在 PR 时验证两个站点，推送 `main` 时通过检查后发布 GitHub Pages；PR 不执行部署。

GitHub Pages 构建使用仓库子路径，Cloudflare 使用根路径，内部链接、订阅和资源地址分别生成。每个站点的 canonical、分享地址和 RSS 使用各自的地址。工作流也兼容 `用户名.github.io` 形式的根站点仓库。

本地验证（需要 Python 3）：

```bash
python3 -m unittest discover -s tests
bash scripts/build-cloudflare.sh /tmp/blog-cloudflare-check
bash scripts/hugo.sh --gc --minify --baseURL https://littlehamsterxu.github.io/blog/ --destination /tmp/blog-github-check
python3 scripts/check-site.py /tmp/blog-github-check --base-url https://littlehamsterxu.github.io/blog/
```

检查涵盖站内链接、锚点、图片、字体、模块依赖、订阅和站点地图；不探测外部网站是否在线。

## 图片与图表

- 本地图片可放在 `static/images/`、`assets/images/` 或文章页面包中；自动输出图片尺寸。宽于 720px 的 JPEG、PNG、WebP 会生成多尺寸 WebP，点击放大仍可查看原图，GIF 保持原格式。
- 图片标题会显示为说明，例如 `![图片替代文字](/images/example.png "图片说明")`。外链图片不会自动下载或转换。
- KaTeX 0.18.4 的样式和字体、Mermaid 11.12.2 的脚本已保存到 `static/vendor/`，只有需要的文章才加载；图表随深浅主题重新绘制。
- 第三方资源保留许可证、来源及 npm 包完整性摘要，见各库目录中的 `SOURCE.json`。更新时需同步模块分片和模板中的版本路径。

## 维护

- 首页和页脚提供 RSS 入口（`/index.xml`），输出文章全文；“关于”页不进入订阅。
- 默认分享图为 `static/images/default-share.png`，通过 `hugo.yaml` 的 `params.images` 配置，仅用于分享信息，不会成为文章列表封面；文章自身的 `cover.image` 优先。
- 分享图可直接替换为同名的 1200 × 630 PNG。重绘当前版本可安装 `Pillow` 后运行 `scripts/create-share-image.py`（使用 Linux 系统的 DejaVu Sans 和 Noto Sans CJK 字体）。
- 字体使用完整字库分片：现有内容常用字和界面文字使用小字库，其余字符按需加载，新增文章无需重新生成。原始字体和 OFL 许可保留在 `static/fonts/`。
- 分片文件和 `assets/css/extended/font-subsets.css` 一起提交，正常构建不需要 Python。需要重新优化常用字时，在本地运行：

  ```bash
  python3 -m venv .local/font-tools
  .local/font-tools/bin/pip install 'fonttools[woff]==4.65.0'
  .local/font-tools/bin/python scripts/subset-font.py
  ```

- `themes/PaperMod` 保存为当前公开快照，不包含主题仓库历史。
- 站点扩展优先放在 `layouts/`、`assets/`、`data/` 和 `content/` 中。
- 新栏目先积累内容，再加入导航。
- 首页介绍修改 `hugo.yaml` 的 `params.homeInfoParams`；旧的未使用首页配置及样式已清理。
