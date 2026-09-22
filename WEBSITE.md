# 网站维护与发布

网站是纯静态 HTML/CSS/JavaScript，无需 Node 构建、第三方字体或 CDN。页面内容来自本地 `main.tex`；作者名单采用项目提供的名单。仅参考 DeltaWAM 的内容组织思路，页面代码独立编写。

页面以演示为主：统一使用参考页的 Inter / 系统无衬线字体栈，白底、靛蓝强调色；标题分两行，ACG-WAM 使用紫色。资源按钮与首屏介绍视频之间保留 80px 间距（手机 56px）。方法部分用两组交错图文展示 Fig. 1 和 Fig. 2 矢量图，配标题和正文。每项真机任务始终并排展示两个视频，积木堆叠和玩具入杯分别展示机器人左臂、右臂。仿真桌面三列、窄屏两列，用全局场景和样例按钮切换，保留 24 段录像的访问。页面不设置指标表、数字卡片、装饰性小字或重复部署说明。

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
- 项目介绍：`ACG-WAM_720p.mp4`，2 分 40.8 秒，约 15.7 MiB。保留原始 H.264 视频和 AAC 音轨，仅调整 MP4 faststart；默认静音自动播放，可以通过播放器开启声音。
- 所有 MP4 均使用 `faststart`，将索引移到文件前部，便于网页渐进播放。视频设置 `autoplay muted loop playsinline`，进入视野时自动播放，离开后暂停，再次进入时恢复；切换仿真样例后继续播放。
- 原始 MOV/MP4 未修改，也未放入仓库。`assets/media-manifest.json` 记录来源相对路径、源文件 SHA-256、压缩前后大小、时长和输出编码。

H.264 不是容器格式；MP4 是容器。与 H.265 相比，H.264 的主要优势是兼容性，这次体积下降主要来自降低分辨率、帧率、码率和去除音轨。只修改扩展名不会压缩视频。

GitHub 普通 Git 单文件超过 50 MiB 会警告，超过 100 MiB 会拒绝；浏览器直接上传单文件限制为 25 MiB。当前最大视频约 15.7 MiB，适合直接随分支上传。GitHub Pages 已发布站点不能超过 1 GB，软带宽限制为每月 100 GB。后续视频明显增多时，可把视频迁移到对象存储/CDN，再替换 `src`；不要将大批 4K 原片提交到 Git 历史。

官方文档：
- [GitHub 文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub Pages 使用限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
- [配置 Pages 发布源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [浏览器视频编码指南](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Video_codecs)

## 更新内容

- 修改页面文字、作者、结果、视频任务：编辑 `tools/build_site.py`，运行 `python tools/build_site.py`。
- 修改样式：`assets/site.css`；样例切换、场景切换与引用复制：`assets/site.js`。修改后运行 `python tools/build_site.py`，更新 HTML 中的资源版本号，避免旧缓存影响展示。
- 重新生成媒体：安装 `imageio-ffmpeg pymupdf pillow`，执行 `python tools/prepare_media.py --source-root C:/path/to/BaiduSyncdisk`。脚本默认跳过已有视频；需要重压某段时先将对应输出 MP4 和 JPG 移出输出目录。脚本针对本次 HLG 原片设计，其他色彩空间需调整转换参数。
- 更新 Fig. 1、Fig. 2 和项目介绍视频：运行 `python tools/prepare_presentation_assets.py --figure PATH_TO_FIGURE_ARCHITECTURE_V4_PDF --video PATH_TO_ACG_WAM_720P_MP4`，Fig. 1 使用同目录的 `teaser_v10.pdf`。SVG 的文字转为路径，线条保持矢量，原始照片保留源像素；图片透明蒙版合并为 PNG alpha，避免渲染器兼容问题。原始 PDF 另存于资源目录，页面不展示 PDF 下载说明。来源摘要和左右臂样例映射见 `assets/presentation-manifest.json`。
- `assets/paper/acg-wam.pdf` 是与 `main.tex` 对应的现有匿名稿。正式作者版或 arXiv 发布后，更新 PDF／链接及 BibTeX。尚无正式发表信息，不标注已被会议录用。
- Model 链接使用项目指定的 `https://huggingface.co/RoboOpus/ACG-WAM`。README 按项目要求仅保留论文、作者、资源链接及引用。

## 本次检查记录

原有 34 段视频已检查完整解码、H.264 像素格式及 MP4 faststart。页面现直接显示 6 段真机录像，另外 4 段保留在资源目录；仿真有 24 种选择。左右臂方向按机器人自身视角标注，通过抽帧核对。Fig. 1、Fig. 2 的导出 SVG 已渲染检查。使用 Playwright 驱动隔离的 Edge 浏览器，检查 1440、1280、1024、768、600、390px 六种宽度的两行标题、视频列数和横向溢出，并截图复核。实际验证介绍视频、真机和仿真自动播放，以及样例切换和滚动返回后的恢复播放。

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
assets/presentation-manifest.json 介绍视频、矢量图及左右臂映射
tools/build_site.py         生成网站 HTML
tools/prepare_media.py      转码、封面和论文插图导出
tools/prepare_presentation_assets.py 导出 Fig. 1、Fig. 2 和介绍视频
.nojekyll                  让 Pages 直接发布静态文件
```
