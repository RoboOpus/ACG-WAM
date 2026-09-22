# 网站维护与发布

网站是纯静态 HTML/CSS/JavaScript，无需 Node 构建、第三方字体或 CDN。页面内容来自本地 `main.tex`；作者名单采用项目提供的名单。参考 DeltaWAM 的方法／实验／视频展示结构，页面代码独立编写。

## 本地预览

在 `website` 分支的仓库根目录运行：

```powershell
python -m http.server 8955 --bind 127.0.0.1
```

浏览器打开 <http://127.0.0.1:8955/>。也可直接打开 `index.html`；复制引用按钮在不支持剪贴板权限时会自动选中文本。

## 发布到 GitHub Pages

1. 将本地 `main` 和 `website` 分支推送到已配置的 `origin`。推送时需通过有仓库写入权限的 GitHub 账号认证。
2. 在 `RoboOpus/ACG-WAM` 的 **Settings → Pages** 中，将 Source 设为 **Deploy from a branch**。
3. 选择 **website** 分支和 **/(root)**，保存，等待 Pages 部署完成。
4. 打开 <https://RoboOpus.github.io/ACG-WAM/> 验证页面和视频。

```powershell
git push -u origin main
git push -u origin website
```

如果要将主页放在 `AIGeeksGroup.github.io/ACG-WAM/`，需要将站点发布到 AIGeeksGroup 名下的相应仓库，或另设重定向。当前远端是 RoboOpus，不能仅修改 HTML 中的链接来改变 Pages 域名。

`main` 分支保存 README 和后续算法代码，`website` 分支保存网站及网页视频。两个分支使用独立历史，避免主分支包含网站视频。日常修改前先用 `git switch website` 切换到网站分支，修改并检查后再提交和推送。

## 视频是怎样上传和播放的

压缩版文件在 `assets/videos/`，跟 HTML 一起通过 `git add`、`git commit`、`git push` 上传。GitHub Pages 直接提供静态 MP4，页面用 `<video>` 播放相对路径。没有额外的视频上传服务，也不需要把视频转为 GIF。

- 真机：10 段，原片约 768 MiB，为 4K / 约 60 fps 的 H.265（HEVC）HLG HDR；网页版本约 73 MiB，为 H.264 / MP4、1920×1080、30 fps、CRF 24、8-bit `yuv420p`。
- 做了 HLG/BT.2020 → SDR/BT.709 色彩转换；去除现场音轨、位置等容器元数据；保持完整时长和原速，未剪辑、未加速。
- 仿真：24 段，约 1 MiB，源文件已经是 H.264、320×240、10 fps，仅重封装，视频流不重新编码。
- 所有 MP4 均使用 `faststart`，将索引移到文件前部，便于网页渐进播放。播放器使用封面图和延迟加载，避免进入页面就下载所有视频。
- 原始 MOV/MP4 未修改，也未放入仓库。`assets/media-manifest.json` 记录来源相对路径、源文件 SHA-256、压缩前后大小、时长和输出编码。

H.264 不是容器格式；MP4 是容器。与 H.265 相比，H.264 的主要优势是兼容性，这次体积下降主要来自降低分辨率、帧率、码率和去除音轨。只修改扩展名不会压缩视频。

GitHub 普通 Git 单文件超过 50 MiB 会警告，超过 100 MiB 会拒绝；浏览器直接上传单文件限制为 25 MiB。当前最大视频约 14 MiB，适合直接随分支上传。GitHub Pages 已发布站点不能超过 1 GB，软带宽限制为每月 100 GB。后续视频明显增多时，可把视频迁移到对象存储/CDN，再替换 `src`；不要将大批 4K 原片提交到 Git 历史。

官方文档：
- [GitHub 文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub Pages 使用限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
- [配置 Pages 发布源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [浏览器视频编码指南](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Video_codecs)

## 更新内容

- 修改页面文字、作者、结果、视频任务：编辑 `tools/build_site.py`，运行 `python tools/build_site.py`。
- 修改样式：`assets/site.css`；样例切换、场景切换与引用复制：`assets/site.js`。
- 重新生成媒体：安装 `imageio-ffmpeg pymupdf pillow`，执行 `python tools/prepare_media.py --source-root C:/path/to/BaiduSyncdisk`。脚本默认跳过已有视频；需要重压某段时先将对应输出 MP4 和 JPG 移出输出目录。脚本针对本次 HLG 原片设计，其他色彩空间需调整转换参数。
- `assets/paper/acg-wam.pdf` 是与 `main.tex` 对应的现有匿名稿。正式作者版或 arXiv 发布后，更新 PDF／链接及 BibTeX。尚无正式发表信息，不标注已被会议录用。
- Model 链接使用项目指定的 `https://huggingface.co/RoboOpus/ACG-WAM`，页面标记为 planned release。代码尚未提供，README 明确记录发布准备状态。

## 本次检查记录

已检查 34 段视频完整解码、H.264 像素格式、MP4 faststart；已检查页面本地资源与 HTTP 访问，以及 34 个样例选择、仿真场景切换、对应成功率／封面／下载路径和引用复制。当前无可连接浏览器，桌面及手机上的实际排版与浏览器播放仍需视觉复核；上述交互检查使用 jsdom，不代替真实浏览器检查。

## 文件结构

```text
index.html                 网站入口
assets/site.css            响应式样式
assets/site.js             交互逻辑
assets/images/             从论文 PDF 导出的图
assets/paper/acg-wam.pdf    现有论文稿
assets/videos/real/        10 段真机视频与封面
assets/videos/simulation/  24 段仿真视频与封面
assets/media-manifest.json 媒体来源和编码记录
tools/build_site.py         生成网站 HTML
tools/prepare_media.py      转码、封面和论文插图导出
.nojekyll                  让 Pages 直接发布静态文件
```
